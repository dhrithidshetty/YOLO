"""EfficientNet-B2 + CBAM + SPPF + BiFPN + ORIGINAL anchor-based YOLOv5.

Place this file as yolov5_effnetb2.py in an ultralytics/yolov5 v7.0 checkout.
Install that checkout's requirements and a compatible torch/torchvision pair.
Suggested legacy environment: Python 3.10, torch 2.5.1, torchvision 0.20.1,
numpy<2, Pillow<10 (install the CUDA build appropriate for your machine).

  python yolov5_effnetb2.py --smoke-test
  python yolov5_effnetb2.py --data /home/ramyads/YOLO/Datasets/o1/data.yaml --device 0

No yolov5lu.pt or ultralytics.YOLO is used. Only EfficientNet-B2 is pretrained.
The neck, custom feature heads and original YOLOv5 Detect head start from scratch.
Keep this module importable next to YOLOv5 when loading saved checkpoints.

Losses are native YOLOv5 (box + objectness + class); this is an architecture
transfer, not a claim that YOLOv5 and YOLOv8 have identical training objectives.
Native YOLOv5 focal loss applies to both class and objectness BCE.
Copy-paste needs polygon annotations; it has no effect for box-only labels.
Epoch schedule is zero-based. Workers=0 is intentional: v7.0's infinite loader
prefetches batches, so parent-only epoch changes are unreliable with workers>0.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import importlib
import sys
import time
import tempfile
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

_SEED = 42
EFFNET_SPECS = {"b2": {"p3": (3, 48), "p4": (5, 120), "p5": (7, 352)}}

class CBAMBlock(nn.Module):
    def __init__(self,ch,r=16):
        super().__init__()
        rc=max(1,ch//r)
        self.mlp=nn.Sequential(nn.Conv2d(ch,rc,1,bias=False),nn.ReLU(inplace=True),nn.Conv2d(rc,ch,1,bias=False))
        self.sig=nn.Sigmoid()
        self.sp=nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False),nn.Sigmoid())
    def forward(self,x):
        x=x*self.sig(self.mlp(x.mean(dim=(2,3),keepdim=True))+self.mlp(x.amax(dim=(2,3),keepdim=True)))
        return x*self.sp(torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],1))

class ConvBNSiLU(nn.Module):
    def __init__(self,ci,co):
        super().__init__()
        self.b=nn.Sequential(nn.Conv2d(ci,co,1,bias=False),nn.BatchNorm2d(co),nn.SiLU(inplace=True))
    def forward(self,x): return self.b(x)

class _CBN(nn.Module):
    def __init__(self,c1,c2,k=1):
        super().__init__()
        self.c=nn.Conv2d(c1,c2,k,padding=k//2,bias=False)
        self.b=nn.BatchNorm2d(c2); self.a=nn.SiLU(inplace=True)
    def forward(self,x): return self.a(self.b(self.c(x)))

class SPPF_(nn.Module):
    def __init__(self,c1,c2,k=5):
        super().__init__()
        c_=c1//2
        self.cv1=_CBN(c1,c_,1); self.cv2=_CBN(c_*4,c2,1)
        self.m=nn.MaxPool2d(k,stride=1,padding=k//2)
    def forward(self,x):
        x=self.cv1(x); y1=self.m(x); y2=self.m(y1)
        return self.cv2(torch.cat([x,y1,y2,self.m(y2)],1))

class BiFPNConv(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.c=nn.Conv2d(ch,ch,3,padding=1,bias=False)
        self.b=nn.BatchNorm2d(ch); self.a=nn.SiLU(inplace=True)
    def forward(self,x): return self.a(self.b(self.c(x)))

class BiFPNLayer(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.w1=nn.Parameter(torch.ones(2)); self.w2=nn.Parameter(torch.ones(2))
        self.w3=nn.Parameter(torch.ones(3)); self.w4=nn.Parameter(torch.ones(2))
        self.c1=BiFPNConv(ch); self.c2=BiFPNConv(ch)
        self.c3=BiFPNConv(ch); self.c4=BiFPNConv(ch)
        self.up=nn.Upsample(scale_factor=2,mode="nearest")
        self.d3=nn.MaxPool2d(3,stride=2,padding=1)
        self.d4=nn.MaxPool2d(3,stride=2,padding=1)
    def _f(self,w,fs):
        w=F.relu(w)+1e-4; w=w/w.sum()
        return sum(w[i]*fs[i] for i in range(len(fs)))
    def forward(self,fs):
        p3,p4,p5=fs
        p4t=self.c1(self._f(self.w1,[p4,self.up(p5)]))
        p3o=self.c2(self._f(self.w2,[p3,self.up(p4t)]))
        p4o=self.c3(self._f(self.w3,[p4,p4t,self.d3(p3o)]))
        p5o=self.c4(self._f(self.w4,[p5,self.d4(p4o)]))
        return [p3o,p4o,p5o]

class BiFPN(nn.Module):
    def __init__(self,ch,nr=2):
        super().__init__()
        self.ls=nn.ModuleList([BiFPNLayer(ch) for _ in range(nr)])
    def forward(self,fs):
        for l in self.ls: fs=l(fs)
        return fs

class DetectionHead(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.cls=nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,bias=False),nn.BatchNorm2d(ch),nn.SiLU(inplace=True),
            nn.Conv2d(ch,ch,3,padding=1,bias=False),nn.BatchNorm2d(ch),nn.SiLU(inplace=True))
        self.reg=nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,bias=False),nn.BatchNorm2d(ch),nn.SiLU(inplace=True))
    def forward(self,x): return self.cls(x)+self.reg(x)

class EffNetFullBiFPNDH(nn.Module):
    out_channels: List[int]
    def __init__(self, model_size='b2', pretrained=True, bifpn_channels=192, cbam_reduction=16, se_reduction=4, bifpn_rounds=2, **_):
        super().__init__()
        ch=bifpn_channels
        w=getattr(models,f'EfficientNet_{model_size.upper()}_Weights').DEFAULT if pretrained else None
        ef=getattr(models,f'efficientnet_{model_size}')(weights=w)
        self.features=nn.Sequential(*list(ef.features.children())[:8])
        arch=EFFNET_SPECS[model_size]
        self.p3i,p3c=arch['p3']; self.p4i,p4c=arch['p4']; self.p5i,p5c=arch['p5']
        self.pp3=ConvBNSiLU(p3c,ch); self.pp4=ConvBNSiLU(p4c,ch); self.pp5=ConvBNSiLU(p5c,ch)
        self.a3=CBAMBlock(ch,cbam_reduction); self.a4=CBAMBlock(ch,cbam_reduction); self.a5=CBAMBlock(ch,cbam_reduction)
        self.bn3=nn.BatchNorm2d(ch); self.bn4=nn.BatchNorm2d(ch); self.bn5=nn.BatchNorm2d(ch)
        self.sppf=SPPF_(ch,ch,5)
        self.bifpn=BiFPN(ch,bifpn_rounds)
        self.dh3=DetectionHead(ch); self.dh4=DetectionHead(ch); self.dh5=DetectionHead(ch)
        self.out_channels=[ch,ch,ch]
    def forward(self,x):
        p3=p4=p5=None
        for i,l in enumerate(self.features):
            x=l(x)
            if   i==self.p3i: p3=self.bn3(self.a3(self.pp3(x)))
            elif i==self.p4i: p4=self.bn4(self.a4(self.pp4(x)))
            elif i==self.p5i: p5=self.bn5(self.a5(self.pp5(x))); p5=self.sppf(p5)
        p3,p4,p5=self.bifpn([p3,p4,p5])
        return [self.dh3(p3),self.dh4(p4),self.dh5(p5)]



@dataclass
class Config:
    data: str = '/home/ramyads/YOLO/Datasets/o1/data.yaml'
    results_root: str = '/home/ramyads/YOLO/results'
    bifpn_channels: int = 192
    bifpn_rounds: int = 2
    cbam_reduction: int = 16
    pretrained: bool = True
    epochs: int = 200
    batch_size: int = 8
    image_size: int = 640
    device: str = '0'
    workers: int = 0
    lr0: float = 0.001
    lrf: float = 0.05
    weight_decay: float = 0.0005
    label_smoothing: float = 0.1
    fl_gamma: float = 1.5
    warmup_epochs: float = 5.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    fliplr: float = 0.5
    translate: float = 0.1
    scale: float = 0.5
    mosaic: float = 1.0
    close_mosaic: int = 30
    copy_paste: float = 0.3
    copy_paste_start_epoch: int = 30
    mixup: float = 0.1
    use_tta: bool = True
    eval_conf: float = 0.001  # Low threshold for mAP, not display confidence.
    test_iou: float = 0.60
    patience: int = 100
    plots: bool = True


ANCHORS = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]


def model_config(nc, cfg):
    return {
        'nc': nc, 'depth_multiple': 1.0, 'width_multiple': 1.0,
        'anchors': deepcopy(ANCHORS), 'custom_effnetb2': True,
        'backbone': [[-1, 1, 'EffNetFullBiFPNDH', [
            'b2', cfg.pretrained, cfg.bifpn_channels,
            cfg.cbam_reduction, 4, cfg.bifpn_rounds]]],
        'head': [[-1, 1, 'Detect', ['nc', 'anchors']]],
    }


def register():
    """Extend parsing only for our exact graph; leave ordinary YOLOv5 intact.

    The feature stack returns a list, passed directly to native Detect. Explicit
    channel metadata avoids the custom _ExtractScale parser problem.
    """
    import models.yolo as yolo
    if getattr(yolo.parse_model, '_effnetb2_registered', False):
        return
    original = yolo.parse_model

    def parse_custom(d, ch):
        if not d.get('custom_effnetb2', False):
            return original(d, ch)
        if ch != [3]:
            raise ValueError('EfficientNet-B2 expects three input channels.')
        if (len(d['backbone']) != 1 or len(d['head']) != 1
                or d['backbone'][0][:3] != [-1, 1, 'EffNetFullBiFPNDH']
                or d['head'][0][:3] != [-1, 1, 'Detect']):
            raise ValueError('Unexpected custom graph; do not silently ignore YAML edits.')
        anchors = d['anchors']
        if not isinstance(anchors, list) or len(anchors) != 3 or any(len(a) != 6 for a in anchors):
            raise ValueError('Provide three anchor pairs at each of three scales.')
        stack = EffNetFullBiFPNDH(*d['backbone'][0][3])
        head = yolo.Detect(nc=d['nc'], anchors=anchors, ch=stack.out_channels)
        for i, layer in enumerate((stack, head)):
            layer.i, layer.f = i, -1
            layer.type = type(layer).__name__
            layer.np = sum(p.numel() for p in layer.parameters())
        return nn.Sequential(stack, head), []

    parse_custom._effnetb2_registered = True
    yolo.parse_model = parse_custom


def hyperparameters(cfg, root):
    with (root / 'data/hyps/hyp.scratch-low.yaml').open() as f:
        hyp = yaml.safe_load(f)
    hyp.update(
        lr0=cfg.lr0, lrf=cfg.lrf, weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs, warmup_momentum=cfg.warmup_momentum,
        warmup_bias_lr=cfg.warmup_bias_lr, fl_gamma=cfg.fl_gamma,
        fliplr=cfg.fliplr, translate=cfg.translate, scale=cfg.scale,
        mosaic=cfg.mosaic, copy_paste=0.0, mixup=cfg.mixup,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0,
        shear=0.0, perspective=0.0, flipud=0.0,
    )
    return hyp


def smoke_test(root):
    """No dataset or pretrained download: shapes, loss/backward, TTA, reload."""
    from models.yolo import Model, Detect
    from utils.loss import ComputeLoss
    register()
    cfg = Config(pretrained=False)
    torch.manual_seed(_SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = Model(model_config(3, cfg))
    assert isinstance(model.model[-1], Detect)
    assert model.stride.tolist() == [8.0, 16.0, 32.0]
    model.eval()
    with torch.no_grad():
        for h, w in ((640, 640), (320, 512)):
            x = torch.zeros(1, 3, h, w)
            features = model.model[0](x)
            assert [tuple(f.shape) for f in features] == [
                (1, 192, h // s, w // s) for s in (8, 16, 32)]
            prediction = model(x)[0]
            assert prediction.shape == (1, 3 * sum((h // s) * (w // s) for s in (8, 16, 32)), 8)
        assert torch.isfinite(model(torch.zeros(1, 3, 256, 256), augment=True)[0]).all()
    model.hyp = hyperparameters(cfg, root)
    model.hyp['label_smoothing'] = cfg.label_smoothing
    model.train()
    outputs = model(torch.randn(2, 3, 128, 128))
    assert [tuple(p.shape) for p in outputs] == [
        (2, 3, 128 // s, 128 // s, 8) for s in (8, 16, 32)]
    targets = torch.tensor([[0, 1, .5, .5, .2, .2], [1, 2, .3, .4, .1, .15]])
    loss, parts = ComputeLoss(model)(outputs, targets)
    assert torch.isfinite(loss).all() and torch.isfinite(parts).all()
    loss.backward()
    grad = model.model[0].features[0][0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    model.eval()
    sample = torch.zeros(1, 3, 128, 128)
    with torch.no_grad(), tempfile.TemporaryDirectory() as td:
        before = model(sample)[0]
        checkpoint = Path(td) / 'smoke.pt'
        torch.save({'model': model}, checkpoint)
        # This checkpoint was just created by this test.
        restored = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
        torch.testing.assert_close(before, restored(sample)[0])
    print('PASS: feature shapes, anchor head, strides, loss/backward, TTA and checkpoint reload.')


def train_model(cfg, root):
    import train as v5train
    import val as v5val
    from utils.callbacks import Callbacks

    if cfg.workers != 0:
        raise ValueError('Use workers=0 for exact epoch augmentation scheduling in this wrapper.')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or ',' in cfg.device:
        raise ValueError('This wrapper supports one GPU or CPU; launch without torchrun.')
    dy = Path(cfg.data).expanduser().resolve()
    with dy.open() as f:
        data = yaml.safe_load(f)
    if not data.get('names'):
        raise ValueError('data.yaml must contain class names.')
    nc = len(data['names'])
    if 'nc' in data and int(data['nc']) != nc:
        raise ValueError('data.yaml nc does not match names.')
    register()
    run = f'exp_v5_effnetb2_fullstack_{os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))}'
    project = Path(cfg.results_root).expanduser().resolve() / dy.parent.name
    run_dir = project / run
    run_dir.mkdir(parents=True, exist_ok=False)
    model_yaml, hyp_yaml = run_dir / 'model.yaml', run_dir / 'hyp_input.yaml'
    for path, content in ((model_yaml, model_config(nc, cfg)),
                          (hyp_yaml, hyperparameters(cfg, root)),
                          (run_dir / 'config.yaml', asdict(cfg))):
        path.write_text(yaml.safe_dump(content, sort_keys=False))

    # Capture the actual training dataset. With workers=0, each sample observes
    # the schedule immediately; no worker-side copy or prefetched stale batch.
    state = {'epoch': -1, 'dataset': None}
    original_loader = v5train.create_dataloader

    def capture_loader(*args, **kwargs):
        loader, dataset = original_loader(*args, **kwargs)
        if kwargs.get('augment', False):
            state['dataset'] = dataset
            if cfg.copy_paste and not any(len(s) for s in dataset.segments):
                print('Box-only labels: native YOLOv5 copy-paste is inactive (requires polygons).')
        return loader, dataset

    def epoch_start():
        state['epoch'] += 1
        epoch, dataset = state['epoch'], state['dataset']
        if dataset is None:
            raise RuntimeError('Training dataset was not captured.')
        closed = cfg.close_mosaic > 0 and epoch >= max(0, cfg.epochs - cfg.close_mosaic)
        dataset.mosaic = bool(cfg.mosaic) and not closed and not dataset.rect
        dataset.hyp['mosaic'] = 0.0 if closed else cfg.mosaic
        dataset.hyp['mixup'] = 0.0 if closed else cfg.mixup
        dataset.hyp['copy_paste'] = (
            cfg.copy_paste if epoch >= cfg.copy_paste_start_epoch and not closed else 0.0)
        if epoch in (0, cfg.copy_paste_start_epoch, max(0, cfg.epochs - cfg.close_mosaic)):
            print(f"Epoch {epoch}: mosaic={dataset.hyp['mosaic']}, "
                  f"mixup={dataset.hyp['mixup']}, copy_paste={dataset.hyp['copy_paste']}")

    callbacks = Callbacks()
    callbacks.register_action('on_train_epoch_start', callback=epoch_start)
    # Parse YOLOv5 defaults without accidentally interpreting this script's CLI.
    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        opt = v5train.parse_opt(known=True)
    finally:
        sys.argv = argv
    settings = dict(
        data=str(dy), cfg=str(model_yaml), hyp=str(hyp_yaml), weights='',
        epochs=cfg.epochs, batch_size=cfg.batch_size, imgsz=cfg.image_size,
        device=cfg.device, workers=cfg.workers, project=str(project), name=run,
        exist_ok=True, optimizer='AdamW', cos_lr=True,
        label_smoothing=cfg.label_smoothing, patience=cfg.patience,
        seed=_SEED, freeze=[0], save_period=-1, noplots=not cfg.plots,
    )
    for key, value in settings.items():
        setattr(opt, key, value)
    v5train.create_dataloader = capture_loader
    try:
        v5train.main(opt, callbacks=callbacks)
    finally:
        v5train.create_dataloader = original_loader

    best = run_dir / 'weights/best.pt'
    if not best.exists():
        raise FileNotFoundError(f'Training did not produce {best}')
    metrics_path = project / 'effnetb2_yolov5_metrics.csv'
    for split in ['val'] + (['test'] if data.get('test') else []):
        use_tta = split == 'test' and cfg.use_tta
        results, _, _ = v5val.run(
            data=str(dy), weights=str(best), task=split,
            imgsz=cfg.image_size, batch_size=cfg.batch_size, device=cfg.device,
            workers=cfg.workers, augment=use_tta, conf_thres=cfg.eval_conf,
            iou_thres=cfg.test_iou, plots=cfg.plots,
            project=str(run_dir), name=f'{split}_tta' if use_tta else split,
            exist_ok=True,
        )
        new = not metrics_path.exists()
        with metrics_path.open('a', newline='') as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(['run', 'split', 'tta', 'precision', 'recall', 'map50', 'map50_95'])
            writer.writerow([run, split, use_tta, *map(float, results[:4])])
    print(f'Done: {best}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--data', default=Config.data)
    parser.add_argument('--results-root', default=Config.results_root)
    parser.add_argument('--device', default=Config.device)
    parser.add_argument('--epochs', type=int, default=Config.epochs)
    parser.add_argument('--batch-size', type=int, default=Config.batch_size)
    parser.add_argument('--no-pretrained', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if not (root / 'models/yolo.py').is_file() or not (root / 'train.py').is_file():
        raise SystemExit('Place yolov5_effnetb2.py in the original ultralytics/yolov5 v7.0 repository root.')
    if args.smoke_test:
        smoke_test(root)
    else:
        train_model(Config(data=args.data, results_root=args.results_root,
                           device=args.device, epochs=args.epochs,
                           batch_size=args.batch_size, pretrained=not args.no_pretrained), root)


if __name__ == '__main__':
    # Save custom classes under an importable module name, never __main__.
    importlib.import_module(Path(__file__).stem).main()
