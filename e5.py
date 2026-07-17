

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random, sys, time, csv, yaml, builtins
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict
from torchvision import models

torch.inference_mode = torch.no_grad
from ultralytics import YOLO

_SEED = 42
random.seed(_SEED); np.random.seed(_SEED)
torch.manual_seed(_SEED); torch.cuda.manual_seed_all(_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

EXP_ID   = "05"
EXP_NAME = "effnetb2_cbam_pan_std"






@dataclass
class Config:
    dataset      : str   = "o1"
    data_root    : str   = "/home/ramyads/YOLO/Datasets"
    results_root : str   = "/home/ramyads/YOLO/results"
    model_size   : str   = "b2"
    pretrained_backbone: bool = True
    bifpn_channels: int  = 192
    bifpn_rounds : int   = 2
    cbam_reduction: int  = 16
    se_reduction  : int  = 4
    epochs       : int   = 200
    batch_size   : int   = 8
    image_size   : int   = 640
    device       : str   = "0"
    workers      : int   = 4
    lr0          : float = 0.001
    lrf          : float = 0.05
    weight_decay : float = 0.0005
    label_smoothing: float = 0.1
    fl_gamma     : float = 1.5
    warmup_epochs  : float = 5.0
    warmup_momentum: float = 0.8
    warmup_bias_lr : float = 0.1
    fliplr       : float = 0.5
    translate    : float = 0.1
    scale        : float = 0.5
    mosaic       : float = 1.0
    close_mosaic : int   = 30
    copy_paste   : float = 0.3
    copy_paste_start_epoch: int = 30
    mixup        : float = 0.1
    use_tta      : bool  = True
    test_conf    : float = 0.25
    test_iou     : float = 0.60
    patience     : int   = 100
    plots        : bool  = True
    verbose      : bool  = False

    def save(self, path):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)






def safe_worker_count(requested):
    slurm_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", 0))
    available  = slurm_cpus if slurm_cpus > 0 else (os.cpu_count() or 4)
    cap        = max(1, available // 2)
    chosen     = min(requested, cap)
    if chosen < requested:
        print(f"  ℹ  workers capped {requested}   {chosen}")
    return chosen






EFFNET_SPECS = {
    "b2": {"p3": (3, 48), "p4": (5, 120), "p5": (7, 352)},
}






class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        rc = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, rc, 1),
            nn.SiLU(inplace=True), nn.Conv2d(rc, channels, 1), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x)


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        rc = max(1, channels // reduction)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, rc, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(rc, channels, 1, bias=False))
        self.ch_sigmoid = nn.Sigmoid()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
    def forward(self, x):
        avg_w = self.shared_mlp(x.mean(dim=(2,3), keepdim=True))
        max_w = self.shared_mlp(x.amax(dim=(2,3), keepdim=True))
        x = x * self.ch_sigmoid(avg_w + max_w)
        sa = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * self.spatial(sa)


class ConvBNSiLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))
    def forward(self, x): return self.block(x)


class _ConvKBNSiLU(nn.Module):
    def __init__(self, c1, c2, k=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, padding=k//2, bias=False)
        self.bn = nn.BatchNorm2d(c2); self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))


class SPPF_(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = _ConvKBNSiLU(c1, c_, 1)
        self.cv2 = _ConvKBNSiLU(c_*4, c2, 1)
        self.m   = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
    def forward(self, x):
        x = self.cv1(x); y1 = self.m(x); y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))


class BiFPNConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(ch); self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))


class BiFPNLayer(nn.Module):
    """NaN-safe fusion: relu(w)+1e-4 before normalise (FIX 2)."""
    def __init__(self, ch):
        super().__init__()
        self.w_td_p4 = nn.Parameter(torch.ones(2))
        self.w_td_p3 = nn.Parameter(torch.ones(2))
        self.w_bu_p4 = nn.Parameter(torch.ones(3))
        self.w_bu_p5 = nn.Parameter(torch.ones(2))
        self.conv_td_p4 = BiFPNConv(ch); self.conv_td_p3 = BiFPNConv(ch)
        self.conv_bu_p4 = BiFPNConv(ch); self.conv_bu_p5 = BiFPNConv(ch)
        self.up   = nn.Upsample(scale_factor=2, mode="nearest")
        self.dn_p3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.dn_p4 = nn.MaxPool2d(3, stride=2, padding=1)
    def _fuse(self, w, feats):
        w = F.relu(w) + 1e-4; w = w / w.sum()
        return sum(w[i] * feats[i] for i in range(len(feats)))
    def forward(self, features):
        p3, p4, p5 = features
        p4_td  = self.conv_td_p4(self._fuse(self.w_td_p4, [p4, self.up(p5)]))
        p3_out = self.conv_td_p3(self._fuse(self.w_td_p3, [p3, self.up(p4_td)]))
        p4_out = self.conv_bu_p4(self._fuse(self.w_bu_p4, [p4, p4_td, self.dn_p3(p3_out)]))
        p5_out = self.conv_bu_p5(self._fuse(self.w_bu_p5, [p5, self.dn_p4(p4_out)]))
        return [p3_out, p4_out, p5_out]


class BiFPN(nn.Module):
    def __init__(self, ch, num_rounds=2):
        super().__init__()
        self.layers = nn.ModuleList([BiFPNLayer(ch) for _ in range(num_rounds)])
    def forward(self, features):
        for layer in self.layers: features = layer(features)
        return features


class DetectionHead(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.cls_branch = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.SiLU(inplace=True))
        self.reg_branch = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.SiLU(inplace=True))
    def forward(self, x): return self.cls_branch(x) + self.reg_branch(x)


class _ExtractScale(nn.Module):
    def __init__(self, out_channels, index):
        super().__init__()
        self.index = index; self.out_channels = out_channels
    def forward(self, x): return x[self.index] if isinstance(x, (list, tuple)) else x






class EfficientNetCBAM(nn.Module):
    out_channels: List[int]
    def __init__(self, model_size="b2", pretrained=True, bifpn_channels=192,
                 cbam_reduction=16, se_reduction=4, **_):
        super().__init__()
        ch = bifpn_channels
        w  = getattr(models, f"EfficientNet_{model_size.upper()}_Weights").DEFAULT if pretrained else None
        effnet = getattr(models, f"efficientnet_{model_size}")(weights=w)
        self.features = nn.Sequential(*list(effnet.features.children()))
        arch = EFFNET_SPECS[model_size]
        self.p3_idx, p3_ch = arch["p3"]; self.p4_idx, p4_ch = arch["p4"]; self.p5_idx, p5_ch = arch["p5"]
        self.proj_p3 = ConvBNSiLU(p3_ch, ch); self.proj_p4 = ConvBNSiLU(p4_ch, ch); self.proj_p5 = ConvBNSiLU(p5_ch, ch)
        self.attn_p3 = CBAMBlock(ch, cbam_reduction); self.attn_p4 = CBAMBlock(ch, cbam_reduction); self.attn_p5 = CBAMBlock(ch, cbam_reduction)
        self.out_channels = [ch, ch, ch]
    def forward(self, x):
        p3 = p4 = p5 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if   i == self.p3_idx: p3 = self.attn_p3(self.proj_p3(x))
            elif i == self.p4_idx: p4 = self.attn_p4(self.proj_p4(x))
            elif i == self.p5_idx: p5 = self.attn_p5(self.proj_p5(x))
        return [p3, p4, p5]






class _FocalBCE(nn.Module):
    def __init__(self, base_bce, gamma=1.5):
        super().__init__()
        self.gamma = gamma; self.reduction = base_bce.reduction
        self.pos_weight = base_bce.pos_weight
        self._bce_none = nn.BCEWithLogitsLoss(reduction="none",
                                               pos_weight=base_bce.pos_weight)
    def forward(self, pred, target):
        loss = self._bce_none(pred, target)
        p_t  = torch.sigmoid(pred)*target + (1-torch.sigmoid(pred))*(1-target)
        loss = (1-p_t)**self.gamma * loss
        return loss.mean() if self.reduction=="mean" else (
               loss.sum()  if self.reduction=="sum"  else loss)

def _get_criterion(trainer):
    return getattr(trainer,"criterion",None) or getattr(trainer,"compute_loss",None)

def _patch_focal(trainer, gamma):
    if gamma <= 0: return
    crit = _get_criterion(trainer)
    for attr in ("bce","bce_cls","bce_obj","cls_loss_fn"):
        node = getattr(crit, attr, None)
        if isinstance(node, nn.BCEWithLogitsLoss):
            setattr(crit, attr, _FocalBCE(node, gamma))
            print(f"     fl_gamma={gamma} patched on criterion.{attr}"); return
    for cn, child in crit.named_children():
        for attr in ("bce","bce_cls","bce_obj","cls_loss_fn"):
            node = getattr(child, attr, None)
            if isinstance(node, nn.BCEWithLogitsLoss):
                setattr(child, attr, _FocalBCE(node, gamma))
                print(f"     fl_gamma={gamma} patched on criterion.{cn}.{attr}"); return
    print("    fl_gamma patch: no BCEWithLogitsLoss found")

def _patch_ls(trainer, ls):
    if ls <= 0: return
    crit = _get_criterion(trainer)
    if hasattr(crit, "label_smoothing"):
        crit.label_smoothing = ls; print(f"     label_smoothing={ls} patched"); return
    for cn, child in crit.named_children():
        if hasattr(child, "label_smoothing"):
            child.label_smoothing = ls
            print(f"     label_smoothing={ls} patched on criterion.{cn}"); return
    try:
        crit.label_smoothing = ls; print(f"     label_smoothing={ls} force-set")
    except Exception as e:
        print(f"     label_smoothing patch failed: {e}")

def make_loss_patch_cb(fl_gamma, label_smoothing):
    patched = {"done": False}
    def _cb(trainer):
        if patched["done"]: return
        print("\n Loss patches (on_before_zero_grad, batch 0)    FIX 1")
        _patch_focal(trainer, fl_gamma); _patch_ls(trainer, label_smoothing)
        patched["done"] = True
    return _cb

def make_copy_paste_cb(start_epoch, val):
    def _cb(trainer):
        if trainer.epoch == start_epoch:
            trainer.args.copy_paste = val
            print(f"\n     copy_paste={val} enabled at epoch {trainer.epoch}    FIX 3")
    return _cb






class MetricsTracker:
    def __init__(self, base_path, dataset_name, run_name, start_time=None):
        self.base_path = Path(base_path); self.dataset_name = dataset_name
        self.run_name  = run_name; self.start_time = start_time or time.time()
        self.all_csv   = self.base_path / "all_models_metrics.csv"
        self.run_dir   = self.base_path / dataset_name / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self.all_csv.exists():
            with open(self.all_csv, "w", newline="") as f:
                csv.writer(f).writerow(["exp_id","exp_name","dataset","map50","map50_95"])
    def update(self, results):
        map50 = map50_95 = 0.0
        if hasattr(results, "box"):
            map50    = float(getattr(results.box,"map50",0.0))
            map50_95 = float(getattr(results.box,"map",  0.0))
        with open(self.all_csv, "a", newline="") as f:
            csv.writer(f).writerow([EXP_ID, EXP_NAME, self.dataset_name, map50, map50_95])
        print(f"     Test mAP50={map50:.4f}  mAP50-95={map50_95:.4f}")






def register_custom_modules():
    import ultralytics.nn.tasks   as tasks_mod
    import ultralytics.nn.modules as modules_mod
    for cls in (SEBlock, CBAMBlock, ConvBNSiLU, _ConvKBNSiLU, SPPF_, BiFPNConv, BiFPNLayer, BiFPN, DetectionHead, _ExtractScale, EfficientNetCBAM,):
        name = cls.__name__
        setattr(modules_mod, name, cls); modules_mod.__dict__[name] = cls
        setattr(tasks_mod,   name, cls); tasks_mod.__dict__[name]   = cls
        setattr(builtins,    name, cls)
    print(f"     Registered: {[c.__name__ for c in (SEBlock, CBAMBlock, ConvBNSiLU, _ConvKBNSiLU, SPPF_, BiFPNConv, BiFPNLayer, BiFPN, DetectionHead, _ExtractScale, EfficientNetCBAM,)]}")



def build_yaml(nc, ch, tag, save_dir):
    """backbone   PAN (C2f)   Std Detect"""
    d = {
        "nc": nc,
        "backbone": [
            [-1, 1, "EfficientNetCBAM", ["b2", True, ch, 16, 4]],
            [ 0, 1, "_ExtractScale", [ch, 0]],
            [ 0, 1, "_ExtractScale", [ch, 1]],
            [ 0, 1, "_ExtractScale", [ch, 2]],
        ],
        "head": [
            [3, 1, "nn.Upsample",    [None, 2, "nearest"]],
            [[4, 2], 1, "Concat",    [1]],
            [-1, 3, "C2f",           [ch]],
            [-1, 1, "nn.Upsample",   [None, 2, "nearest"]],
            [[7, 1], 1, "Concat",    [1]],
            [-1, 3, "C2f",           [ch]],       
            [-1, 1, "Conv",          [ch, 3, 2]],
            [[10, 6], 1, "Concat",   [1]],
            [-1, 3, "C2f",           [ch]],       
            [-1, 1, "Conv",          [ch, 3, 2]],
            [[13, 3], 1, "Concat",   [1]],
            [-1, 3, "C2f",           [ch]],       
            [[9, 12, 15], 1, "Detect", [nc]],
        ],
    }
    p = Path(save_dir) / f"{tag}.yaml"
    with open(p, "w") as f: yaml.dump(d, f, default_flow_style=False, sort_keys=False)
    return p



DESCRIPTION = """  Backbone: EffNet-B2 + Proj + Full CBAM (ch+sp)   SPPF: NONE
  Neck: PAN   Head: Std"""


def main():
    total_start = time.time()
    cfg = Config()
    cfg.workers = safe_worker_count(cfg.workers)
    data_yaml    = Path(cfg.data_root) / cfg.dataset / "data.yaml"
    results_base = Path(cfg.results_root)
    results_base.mkdir(parents=True, exist_ok=True)
    if not data_yaml.exists(): raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    with open(data_yaml) as f: nc = yaml.safe_load(f).get("nc", 12)

    print("="*65); print(f"  EXP #{EXP_ID} — {EXP_NAME.upper()}"); print(DESCRIPTION); print("="*65)

    register_custom_modules()
    yaml_path = build_yaml(nc, cfg.bifpn_channels, EXP_NAME, results_base)

    job_id   = os.environ.get("SLURM_JOB_ID", f"local_{int(time.time())}")
    run_name = f"exp{EXP_ID}_{EXP_NAME}_{job_id}"
    tracker  = MetricsTracker(results_base, cfg.dataset, run_name, total_start)
    cfg.save(tracker.run_dir / "config.yaml")

    model = YOLO(str(yaml_path))
    for p in model.model.parameters(): p.data = p.data.clone()
    model.add_callback("on_before_zero_grad", make_loss_patch_cb(cfg.fl_gamma, cfg.label_smoothing))
    model.add_callback("on_train_epoch_start", make_copy_paste_cb(cfg.copy_paste_start_epoch, cfg.copy_paste))

    model.train(**{**dict(
        data=str(data_yaml), epochs=cfg.epochs, imgsz=cfg.image_size,
        batch=cfg.batch_size, device=cfg.device, workers=cfg.workers,
        project=str(results_base/cfg.dataset), name=run_name,
        exist_ok=True, save=True, save_period=0, verbose=cfg.verbose, plots=cfg.plots, freeze=0,
        optimizer="AdamW", lr0=cfg.lr0, lrf=cfg.lrf, weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs, warmup_momentum=cfg.warmup_momentum, warmup_bias_lr=cfg.warmup_bias_lr,
        fliplr=cfg.fliplr, translate=cfg.translate, scale=cfg.scale, mosaic=cfg.mosaic, close_mosaic=cfg.close_mosaic,
        copy_paste=0.0, mixup=cfg.mixup,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0,
        patience=cfg.patience, cos_lr=True, seed=_SEED)})

    best = results_base/cfg.dataset/run_name/"weights"/"best.pt"
    register_custom_modules()
    bm = YOLO(str(best))
    bm.val(data=str(data_yaml), imgsz=cfg.image_size, batch=cfg.batch_size, device=cfg.device, split="val")
    test_dir = Path(cfg.data_root)/cfg.dataset/"images"/"test"
    if test_dir.exists():
        res = bm.val(data=str(data_yaml), split="test", imgsz=cfg.image_size,
                     batch=cfg.batch_size, device=cfg.device, augment=cfg.use_tta)
        tracker.update(res)
    print(f"\n\u2713 Exp #{EXP_ID} done in {(time.time()-total_start)/3600:.2f} h     {tracker.run_dir}")

if __name__ == "__main__":
    main()
