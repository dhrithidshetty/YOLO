"""EfficientNet-B2 + projections + CBAM + SPPF(P5) + BiFPN x2 + feature heads.
The final detection head and losses come from the named model's own framework.
Only EfficientNet-B2 uses pretrained weights; detection layers train from scratch.
Keep this Python module importable to reload checkpoints. This custom hybrid is
not the stock L model: its native backbone and neck have been replaced.

YOLO11: install ultralytics==8.3.0 and compatible torch/torchvision.
Recommended reproducible runtime: Python 3.10, torch==2.5.1,
torchvision==0.20.1, numpy<2. Use the torch build for your CUDA installation.

  python yolo11_effnetb2.py --smoke-test
  python yolo11_effnetb2.py --data /home/ramyads/YOLO/Datasets/o1/data.yaml --device 0

Uses the YOLO11-era Detect head (depthwise classification convolutions) and native
box/classification/DFL loss. Do not describe this as a distinct stock YOLO11-L.
The supplied loss callbacks were removed: setting an unused smoothing attribute
is not label smoothing. No custom focal/smoothing modification is claimed.
Copy-paste is disabled for this box-detection workflow. Mosaic closes natively.
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
    patience: int = 100
    use_tta: bool = True


def require_version():
    import ultralytics
    if ultralytics.__version__ != '8.3.0':
        raise RuntimeError('This adapter targets ultralytics==8.3.0; install that version in a separate environment.')


def architecture(nc, cfg):
    return dict(nc=nc, custom_effnet_yolo11=True,
                backbone=[[-1, 1, 'EffNetFullBiFPNDH',
                           ['b2', cfg.pretrained, cfg.channels, cfg.cbam_reduction, 4, cfg.rounds]]],
                head=[[-1, 1, 'Detect', [nc]]])


def register():
    require_version()
    import ultralytics.nn.tasks as tasks
    from ultralytics.nn.modules import Detect
    if getattr(tasks.parse_model, '_effnet_yolo11', False):
        return
    original = tasks.parse_model

    def parse_custom(d, ch, verbose=True):
        if not d.get('custom_effnet_yolo11', False):
            return original(d, ch, verbose)
        if ch != 3 or len(d['backbone']) != 1 or len(d['head']) != 1:
            raise ValueError('Expected the three-channel EfficientNet/Detect graph.')
        if d['backbone'][0][:3] != [-1, 1, 'EffNetFullBiFPNDH'] or d['head'][0][:3] != [-1, 1, 'Detect']:
            raise ValueError('Unexpected custom architecture.')
        stack = EffNetFullBiFPNDH(*d['backbone'][0][3])
        head = Detect(nc=d['nc'], ch=stack.out_channels)
        for i, layer in enumerate((stack, head)):
            layer.i, layer.f, layer.type = i, -1, type(layer).__name__
            layer.np = sum(p.numel() for p in layer.parameters())
        return nn.Sequential(stack, head), []

    parse_custom._effnet_yolo11 = True
    tasks.parse_model = parse_custom


def smoke_test():
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules import DWConv
    from ultralytics.cfg import get_cfg
    register()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(42)
    model = DetectionModel(architecture(3, Config(pretrained=False)), verbose=False)
    assert model.stride.tolist() == [8., 16., 32.]
    assert isinstance(model.model[-1].cv3[0][0][0], DWConv)
    model.eval()
    with torch.no_grad():
        for h, w in ((640, 640), (320, 512)):
            x = torch.zeros(1, 3, h, w)
            fs = model.model[0](x)
            assert [tuple(f.shape) for f in fs] == [(1, 192, h//s, w//s) for s in (8, 16, 32)]
            assert model(x)[0].shape == (1, 7, sum((h//s)*(w//s) for s in (8, 16, 32)))
        assert torch.isfinite(model(torch.zeros(1, 3, 256, 256), augment=True)[0]).all()
    model.args = get_cfg()
    model.train()
    batch = dict(img=torch.randn(2, 3, 128, 128), batch_idx=torch.tensor([0, 1]),
                 cls=torch.tensor([[1.], [2.]]),
                 bboxes=torch.tensor([[.5, .5, .2, .2], [.3, .4, .1, .15]]))
    loss, items = model(batch)
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    grad = model.model[0].features[0][0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    # Check that the framework optimizer includes the scalar fusion parameters.
    from ultralytics.engine.trainer import BaseTrainer
    optimizer = BaseTrainer.build_optimizer(None, model, name='AdamW', lr=.001)
    grouped = {id(p) for g in optimizer.param_groups for p in g['params']}
    assert all(id(p) in grouped for p in model.parameters() if p.requires_grad)
    model.eval()
    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad(), tempfile.TemporaryDirectory() as td:
        before = model(x)[0]
        path = Path(td) / 'model.pt'
        torch.save({'model': model}, path)
        restored = torch.load(path, map_location='cpu', weights_only=False)['model']
        torch.testing.assert_close(before, restored(x)[0])
    # Exercise the same YAML entry and reconstruction path used by YOLO.train().
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as td:
        yp = Path(td) / 'effnet_yolo11.yaml'
        yp.write_text(yaml.safe_dump(architecture(3, Config(pretrained=False))))
        wrapper = YOLO(str(yp), task='detect')
        rebuilt = DetectionTrainer.get_model(SimpleNamespace(data={'nc': 4}),
                                              cfg=wrapper.model.yaml, verbose=False)
        assert rebuilt.model[-1].nc == 4 and rebuilt.stride.tolist() == [8., 16., 32.]
    print('PASS: shapes, YOLO11 head, native loss/backward, optimizer coverage, TTA, checkpoint reload, trainer rebuild.')


def train_model(cfg):
    from ultralytics import YOLO
    register()
    if ',' in cfg.device or int(os.getenv('WORLD_SIZE', '1')) != 1:
        raise ValueError('Use one GPU or CPU; this wrapper is not configured for DDP subprocesses.')
    data_path = Path(cfg.data).expanduser().resolve()
    data = yaml.safe_load(data_path.read_text())
    nc = len(data['names'])
    if 'nc' in data and int(data['nc']) != nc:
        raise ValueError('Dataset nc disagrees with names.')
    project = Path(cfg.results_root).expanduser().resolve() / data_path.parent.name
    run = 'yolo11_effnetb2_' + time.strftime('%Y%m%d_%H%M%S')
    run_dir = project / run
    run_dir.mkdir(parents=True, exist_ok=False)
    yp = run_dir / 'effnetb2_yolo11.yaml'
    yp.write_text(yaml.safe_dump(architecture(nc, cfg), sort_keys=False))
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(asdict(cfg)))
    model = YOLO(str(yp), task='detect')
    model.train(
        data=str(data_path), epochs=cfg.epochs, batch=cfg.batch, imgsz=cfg.imgsz,
        device=cfg.device, workers=cfg.workers, project=str(project), name=run, exist_ok=True,
        optimizer='AdamW', lr0=cfg.lr0, lrf=cfg.lrf, weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs, warmup_momentum=cfg.warmup_momentum,
        warmup_bias_lr=cfg.warmup_bias_lr, cos_lr=True, seed=42, freeze=0,
        mosaic=1.0, close_mosaic=cfg.close_mosaic, mixup=.1, copy_paste=0.,
        fliplr=.5, translate=.1, scale=.5, hsv_h=0., hsv_s=0., hsv_v=0.,
        degrees=0., shear=0., perspective=0., flipud=0., patience=cfg.patience,
        save=True, save_period=-1, plots=True,
    )
    best = YOLO(str(run_dir / 'weights/best.pt'), task='detect')
    with (run_dir / 'evaluation.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'tta', 'map50', 'map50_95'])
        for split in ['val'] + (['test'] if data.get('test') else []):
            tta = split == 'test' and cfg.use_tta
            result = best.val(data=str(data_path), split=split, imgsz=cfg.imgsz,
                              batch=cfg.batch, device=cfg.device, augment=tta,
                              conf=.001, iou=.6, project=str(run_dir), name=split,
                              exist_ok=True)
            writer.writerow([split, tta, float(result.box.map50), float(result.box.map)])
    print(f'Done: {run_dir}')


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
    if args.smoke_test:
        smoke_test()
    else:
        train_model(Config(data=args.data, results_root=args.results_root, device=args.device,
                           epochs=args.epochs, batch=args.batch_size, pretrained=not args.no_pretrained))


if __name__ == '__main__':
    importlib.import_module(Path(__file__).stem).main()
