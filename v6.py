"""EfficientNet-B2 + projections + CBAM + SPPF(P5) + BiFPN x2 + feature heads.
The final detection head and losses come from the named model's own framework.
Only EfficientNet-B2 uses pretrained weights; detection layers train from scratch.
Keep this Python module importable to reload checkpoints. This custom hybrid is
not the stock L model: its native backbone and neck have been replaced.

YOLOv6: put yolov6_effnetb2.py at the root of meituan/YOLOv6 tag 0.4.0.
Run from that directory after installing its requirements. Recommended legacy
runtime: Python 3.10, torch==2.5.1, torchvision==0.20.1, numpy<2, Pillow<10.
Use the torch build suitable for your CUDA installation. The upstream training
checkpoint code predates torch 2.6's weights_only default.

  python yolov6_effnetb2.py --smoke-test
  python yolov6_effnetb2.py --data /home/ramyads/YOLO/Datasets/o1/data.yaml --device 0

Uses YOLOv6 anchor-free EffiDeHead, DFL/GIoU/VarifocalLoss, and native trainer.
Auxiliary anchor-based training and distillation are disabled. Native augmentation
supports mosaic/mixup, closed for the last 30 epochs. No copy-paste, label-smoothing,
user focal-gamma override, TTA or early stopping is claimed. Standard evaluation
uses the built-in PR mAP calculation, so COCO JSON annotations are not required.
AdamW grouping explicitly includes BiFPN fusion weights, which the original
YOLOv6 optimizer's weight/bias-only collector would miss.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import csv
import importlib
import sys
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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
    epochs: int = 200
    batch: int = 8
    imgsz: int = 640
    device: str = '0'
    workers: int = 4
    pretrained: bool = True
    channels: int = 192
    rounds: int = 2
    cbam_reduction: int = 16
    lr0: float = .001
    lrf: float = .05
    weight_decay: float = .0005
    warmup_epochs: float = 5.
    warmup_momentum: float = .8
    warmup_bias_lr: float = .1
    close_mosaic: int = 30


def architecture(root, options):
    from yolov6.utils.config import Config as V6Config
    cfg = V6Config.fromfile(str(root / 'configs/yolov6l.py'))
    cfg.model.type = 'YOLOv6_effnetb2'
    cfg.model.custom_effnetb2 = True
    cfg.model.pretrained = None  # Never load stock YOLOv6 weights into this graph.
    cfg.model.backbone = dict(type='EffNetFullBiFPNDH', model_size='b2',
                              pretrained=options.pretrained, bifpn_channels=options.channels,
                              cbam_reduction=options.cbam_reduction, bifpn_rounds=options.rounds)
    cfg.model.neck = dict(type='Identity', note='BiFPN and custom feature heads are inside backbone')
    cfg.model.head.in_channels = [options.channels] * 3
    cfg.model.head.anchors = 1
    cfg.model.head.use_dfl = True
    cfg.model.head.reg_max = 16
    cfg.model.head.atss_warmup_epoch = 0
    cfg.solver.optim = 'AdamW'
    cfg.solver.lr0 = options.lr0
    cfg.solver.lrf = options.lrf
    cfg.solver.weight_decay = options.weight_decay
    cfg.solver.warmup_epochs = options.warmup_epochs
    cfg.solver.warmup_momentum = options.warmup_momentum
    cfg.solver.warmup_bias_lr = options.warmup_bias_lr
    cfg.solver.momentum = .9  # AdamW beta1
    cfg.data_aug.update(dict(hsv_h=0., hsv_s=0., hsv_v=0., degrees=0.,
                             translate=.1, scale=.5, shear=0., flipud=0., fliplr=.5,
                             mosaic=1., mixup=.1))
    cfg.eval_params = dict(conf_thres=.001, do_coco_metric=False, do_pr_metric=True,
                           plot_curve=True, plot_confusion_matrix=False)
    return cfg


def register():
    import yolov6.models.yolo as yolo
    if getattr(yolo.build_network, '_effnetb2', False):
        return
    original = yolo.build_network

    def build_custom(cfg, channels, num_classes, num_layers, fuse_ab=False, distill_ns=False):
        if not cfg.model.get('custom_effnetb2', False):
            return original(cfg, channels, num_classes, num_layers, fuse_ab, distill_ns)
        if channels != 3 or num_layers != 3 or fuse_ab or distill_ns:
            raise ValueError('This adapter supports RGB, P3/P4/P5 and the standard anchor-free head only.')
        from yolov6.models.effidehead import Detect, build_effidehead_layer
        params = dict(cfg.model.backbone)
        params.pop('type')
        stack = EffNetFullBiFPNDH(**params)
        # The upstream builder indexes 6, 8, 10 in its architecture channel list.
        channel_list = [stack.out_channels[0]] * 11
        layers = build_effidehead_layer(channel_list, 1, num_classes, reg_max=16, num_layers=3)
        head = Detect(num_classes=num_classes, num_layers=3, head_layers=layers,
                      use_dfl=True, reg_max=16)
        return stack, nn.Identity(), head

    build_custom._effnetb2 = True
    yolo.build_network = build_custom


def build_adamw(cfg, model):
    """Include every trainable parameter exactly once, including BiFPN w1..w4.

    Keep native group order: no-decay, decay, biases. YOLOv6 warms up group 2
    using warmup_bias_lr, so changing this order would change training behavior.
    """
    no_decay, decay, biases = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith('.bias'):
            biases.append(parameter)
        elif parameter.ndim < 2:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW(no_decay, lr=cfg.solver.lr0,
                                  betas=(cfg.solver.momentum, .999), weight_decay=0.)
    optimizer.add_param_group(dict(params=decay, weight_decay=cfg.solver.weight_decay))
    optimizer.add_param_group(dict(params=biases, weight_decay=0.))
    grouped = [id(p) for g in optimizer.param_groups for p in g['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(grouped) == len(set(grouped)) and set(grouped) == expected
    return optimizer


def smoke_test(root):
    from yolov6.models.yolo import build_model
    from yolov6.models.effidehead import Detect
    from yolov6.models.losses.loss import ComputeLoss
    register()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(42)
    cfg = architecture(root, Config(pretrained=False))
    model = build_model(cfg, 3, torch.device('cpu'))
    assert isinstance(model.detect, Detect) and model.detect.reg_max == 16
    assert model.stride.tolist() == [8, 16, 32]
    model.eval()
    with torch.no_grad():
        for h, w in ((640, 640), (320, 512)):
            x = torch.zeros(1, 3, h, w)
            fs = model.backbone(x)
            assert [tuple(f.shape) for f in fs] == [(1, 192, h//s, w//s) for s in (8, 16, 32)]
            assert model(x)[0].shape == (1, sum((h//s)*(w//s) for s in (8, 16, 32)), 8)
    criterion = ComputeLoss(num_classes=3, ori_img_size=128, warmup_epoch=0, use_dfl=True, reg_max=16)
    optimizer = build_adamw(cfg, model)
    model.train()
    targets = torch.tensor([[0, 1, .5, .5, .2, .2], [1, 2, .3, .4, .1, .15]])
    # Native YOLOv6 initializes final prediction weights to zero. After the first
    # update those weights transmit gradients to the backbone; check step two.
    for step in range(2):
        optimizer.zero_grad()
        predictions, _ = model(torch.randn(2, 3, 128, 128))
        loss, items = criterion(predictions, targets, epoch_num=1, step_num=step)
        assert torch.isfinite(loss).all() and torch.isfinite(items).all()
        loss.backward()
        if step == 1:
            grad = model.backbone.features[0][0].weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        optimizer.step()
    model.eval()
    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad(), tempfile.TemporaryDirectory() as td:
        before = model(x)[0]
        path = Path(td) / 'model.pt'
        torch.save({'model': model}, path)
        restored = torch.load(path, map_location='cpu', weights_only=False)['model']
        torch.testing.assert_close(before, restored(x)[0])
    print('PASS: shapes, native YOLOv6 head/loss, backbone gradients, optimizer coverage, checkpoint reload.')


def resolve_data(path):
    """Convert an Ultralytics-style data YAML into absolute YOLOv6 directories."""
    source = Path(path).expanduser().resolve()
    data = yaml.safe_load(source.read_text())
    names = data.get('names')
    if isinstance(names, dict):
        keys = sorted(int(k) for k in names)
        if keys != list(range(len(keys))):
            raise ValueError('Class name keys must be contiguous starting at zero.')
        names = [names[k] if k in names else names[str(k)] for k in keys]
    if not names:
        raise ValueError('data.yaml must define class names.')
    if 'nc' in data and int(data['nc']) != len(names):
        raise ValueError('nc differs from the number of names.')
    base = Path(data.get('path') or source.parent).expanduser()
    if not base.is_absolute():
        base = source.parent / base
    result = dict(nc=len(names), names=names, is_coco=False)
    for split in ('train', 'val', 'test'):
        if not data.get(split):
            if split != 'test':
                raise ValueError(f'Missing {split} directory in data.yaml.')
            continue
        if not isinstance(data[split], str):
            raise ValueError(f'YOLOv6 0.4.0 requires one image directory for {split}.')
        directory = Path(data[split]).expanduser()
        directory = (directory if directory.is_absolute() else base / directory).resolve()
        if not directory.is_dir():
            raise ValueError(f'{split} must point to an existing image directory: {directory}')
        result[split] = str(directory)
    return source, result


def train_model(root, options):
    import json
    import pprint
    from yolov6.core import engine
    from yolov6.utils.envs import select_device, set_random_seed
    from tools.train import get_args_parser
    from tools import eval as v6eval
    if ',' in options.device or int(os.getenv('WORLD_SIZE', '1')) != 1:
        raise ValueError('This wrapper supports one GPU or CPU, without DDP.')
    if tuple(int(p) for p in torch.__version__.split('+')[0].split('.')[:2]) >= (2, 6):
        raise RuntimeError('YOLOv6 0.4.0 checkpoint loading requires this legacy setup: torch==2.5.1 and torchvision==0.20.1.')
    register()
    source, data = resolve_data(options.data)
    project = Path(options.results_root).expanduser().resolve() / source.parent.name
    run = 'yolov6_effnetb2_' + time.strftime('%Y%m%d_%H%M%S')
    run_dir = project / run
    run_dir.mkdir(parents=True, exist_ok=False)
    data_path = run_dir / 'data_absolute.yaml'
    data_path.write_text(yaml.safe_dump(data))
    cfg = architecture(root, options)
    cfg_path = run_dir / 'model_config.py'
    plain_cfg = json.loads(json.dumps(cfg._cfg_dict))
    cfg_path.write_text('\n\n'.join(k + ' = ' + pprint.pformat(v) for k, v in plain_cfg.items()) + '\n')
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(asdict(options)))
    args = get_args_parser().parse_args([])
    for key, value in dict(data_path=str(data_path), conf_file=str(cfg_path),
                          epochs=options.epochs, batch_size=options.batch, img_size=options.imgsz,
                          workers=options.workers, device=options.device, output_dir=str(project),
                          name=run, save_dir=str(run_dir), rank=-1, local_rank=-1, world_size=1,
                          eval_interval=1, stop_aug_last_n_epoch=options.close_mosaic,
                          bs_per_gpu=options.batch, fuse_ab=False, distill=False).items():
        setattr(args, key, value)
    # bs_per_gpu=batch keeps native LR scaling at 1, preserving lr0=.001.
    (run_dir / 'args.yaml').write_text(yaml.safe_dump(vars(args)))
    device = select_device(options.device)
    set_random_seed(42, deterministic=True)
    original_optimizer = engine.build_optimizer
    engine.build_optimizer = build_adamw
    try:
        trainer = engine.Trainer(args, cfg, device)
        trainer.train()
    finally:
        engine.build_optimizer = original_optimizer
    best = run_dir / 'weights/best_ckpt.pt'
    if not best.exists():
        raise FileNotFoundError(f'No best checkpoint was produced: {best}')
    with (run_dir / 'evaluation.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'tta', 'map50', 'map50_95'])
        for split in ['val'] + (['test'] if data.get('test') else []):
            result, _, _ = v6eval.run(
                data=str(data_path), weights=str(best), task=split, device=options.device,
                batch_size=options.batch, img_size=options.imgsz, conf_thres=.001,
                iou_thres=.6, save_dir=str(run_dir), name=split,
                do_coco_metric=False, do_pr_metric=True, plot_curve=True,
            )
            writer.writerow([split, False, float(result[0]), float(result[1])])
    print(f'Done: {best}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--data', default=Config.data)
    parser.add_argument('--results-root', default=Config.results_root)
    parser.add_argument('--device', default=Config.device)
    parser.add_argument('--epochs', type=int, default=Config.epochs)
    parser.add_argument('--batch-size', type=int, default=Config.batch)
    parser.add_argument('--no-pretrained', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if not (root / 'yolov6/models/yolo.py').is_file():
        raise SystemExit('Place this file in the meituan/YOLOv6 0.4.0 repository root.')
    if args.smoke_test:
        smoke_test(root)
    else:
        train_model(root, Config(data=args.data, results_root=args.results_root, device=args.device,
                                epochs=args.epochs, batch=args.batch_size, pretrained=not args.no_pretrained))


if __name__ == '__main__':
    importlib.import_module(Path(__file__).stem).main()
