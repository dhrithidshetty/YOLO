

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random, time, csv, yaml, builtins
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

torch.inference_mode = torch.no_grad
from ultralytics import YOLO

_SEED = 42
random.seed(_SEED); np.random.seed(_SEED)
torch.manual_seed(_SEED); torch.cuda.manual_seed_all(_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

EXP_ID   = "08"
EXP_NAME = "yolov8l_pan_dethead"




@dataclass
class Config:
    dataset      : str   = "o1"
    data_root    : str   = "/home/ramyads/YOLO/Datasets"
    results_root : str   = "/home/ramyads/YOLO/results"
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
        with open(path, "w") as f: yaml.dump(asdict(self), f, default_flow_style=False)

def safe_worker_count(requested):
    slurm_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", 0))
    available  = slurm_cpus if slurm_cpus > 0 else (os.cpu_count() or 4)
    cap        = max(1, available // 2)
    chosen     = min(requested, cap)
    if chosen < requested: print(f"  ℹ  workers capped {requested}   {chosen}")
    return chosen




class DetectionHead(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.cls_branch = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.SiLU(inplace=True))
        self.reg_branch = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.SiLU(inplace=True))
    def forward(self, x): return self.cls_branch(x) + self.reg_branch(x)




class _FocalBCE(nn.Module):
    def __init__(self, base_bce, gamma=1.5):
        super().__init__()
        self.gamma=gamma; self.reduction=base_bce.reduction; self.pos_weight=base_bce.pos_weight
        self._bce_none=nn.BCEWithLogitsLoss(reduction="none", pos_weight=base_bce.pos_weight)
    def forward(self, pred, target):
        loss=self._bce_none(pred,target)
        p_t=torch.sigmoid(pred)*target+(1-torch.sigmoid(pred))*(1-target)
        loss=(1-p_t)**self.gamma*loss
        return loss.mean() if self.reduction=="mean" else (loss.sum() if self.reduction=="sum" else loss)

def _get_criterion(trainer):
    return getattr(trainer,"criterion",None) or getattr(trainer,"compute_loss",None)

def _patch_focal(trainer, gamma):
    if gamma<=0: return
    crit=_get_criterion(trainer)
    for attr in ("bce","bce_cls","bce_obj","cls_loss_fn"):
        node=getattr(crit,attr,None)
        if isinstance(node,nn.BCEWithLogitsLoss):
            setattr(crit,attr,_FocalBCE(node,gamma)); print(f"     fl_gamma={gamma} patched on criterion.{attr}"); return
    for cn,child in crit.named_children():
        for attr in ("bce","bce_cls","bce_obj","cls_loss_fn"):
            node=getattr(child,attr,None)
            if isinstance(node,nn.BCEWithLogitsLoss):
                setattr(child,attr,_FocalBCE(node,gamma)); print(f"     fl_gamma={gamma} patched on criterion.{cn}.{attr}"); return
    print("    fl_gamma patch: no BCEWithLogitsLoss found")

def _patch_ls(trainer, ls):
    if ls<=0: return
    crit=_get_criterion(trainer)
    if hasattr(crit,"label_smoothing"): crit.label_smoothing=ls; print(f"     label_smoothing={ls} patched"); return
    for cn,child in crit.named_children():
        if hasattr(child,"label_smoothing"):
            child.label_smoothing=ls; print(f"     label_smoothing={ls} patched on criterion.{cn}"); return
    try: crit.label_smoothing=ls; print(f"     label_smoothing={ls} force-set")
    except Exception as e: print(f"     label_smoothing patch failed: {e}")

def make_loss_patch_cb(fl_gamma, label_smoothing):
    patched={"done":False}
    def _cb(trainer):
        if patched["done"]: return
        print("\n Loss patches (on_before_zero_grad, batch 0)    FIX 1")
        _patch_focal(trainer,fl_gamma); _patch_ls(trainer,label_smoothing)
        patched["done"]=True
    return _cb

def make_copy_paste_cb(start_epoch, val):
    def _cb(trainer):
        if trainer.epoch==start_epoch:
            trainer.args.copy_paste=val
            print(f"\n     copy_paste={val} enabled at epoch {trainer.epoch}    FIX 3")
    return _cb




class MetricsTracker:
    def __init__(self, base_path, dataset_name, run_name, start_time=None):
        self.base_path=Path(base_path); self.dataset_name=dataset_name
        self.run_name=run_name; self.start_time=start_time or time.time()
        self.all_csv=self.base_path/"all_models_metrics.csv"
        self.run_dir=self.base_path/dataset_name/run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self.all_csv.exists():
            with open(self.all_csv,"w",newline="") as f:
                csv.writer(f).writerow(["exp_id","exp_name","dataset","map50","map50_95"])
    def update(self, results):
        map50=map50_95=0.0
        if hasattr(results,"box"):
            map50=float(getattr(results.box,"map50",0.0)); map50_95=float(getattr(results.box,"map",0.0))
        with open(self.all_csv,"a",newline="") as f:
            csv.writer(f).writerow([EXP_ID,EXP_NAME,self.dataset_name,map50,map50_95])
        print(f"     Test mAP50={map50:.4f}  mAP50-95={map50_95:.4f}")




def register_custom_modules():
    import ultralytics.nn.tasks   as tasks_mod
    import ultralytics.nn.modules as modules_mod
    for cls in (DetectionHead,):
        name=cls.__name__
        setattr(modules_mod,name,cls); modules_mod.__dict__[name]=cls
        setattr(tasks_mod,  name,cls); tasks_mod.__dict__[name]=cls
        setattr(builtins,   name,cls)
    print("   Custom modules registered (DetectionHead)")





def build_yaml(nc, tag, save_dir):
    d = {
        "nc": nc,
        "backbone": [
            [-1, 1, "Conv",  [64, 3, 2]],    
            [-1, 1, "Conv",  [128, 3, 2]],   
            [-1, 3, "C2f",   [128, True]],
            [-1, 1, "Conv",  [256, 3, 2]],   
            [-1, 6, "C2f",   [256, True]],   
            [-1, 1, "Conv",  [512, 3, 2]],   
            [-1, 6, "C2f",   [512, True]],   
            [-1, 1, "Conv",  [512, 3, 2]],   
            [-1, 3, "C2f",   [512, True]],
            [-1, 1, "SPPF",  [512, 5]],      
        ],
        "head": [
            [-1, 1, "nn.Upsample",  [None, 2, "nearest"]],   
            [[10, 6], 1, "Concat",  [1]],                     
            [-1, 3, "C2f",          [512]],                   
            [-1, 1, "nn.Upsample",  [None, 2, "nearest"]],   
            [[13, 4], 1, "Concat",  [1]],                     
            [-1, 3, "C2f",          [256]],                   
            [-1, 1, "Conv",         [256, 3, 2]],             
            [[16, 12], 1, "Concat", [1]],                     
            [-1, 3, "C2f",          [512]],                   
            [-1, 1, "Conv",         [512, 3, 2]],             
            [[19, 9], 1, "Concat",  [1]],                     
            [-1, 3, "C2f",          [512]],                   
            
            [15, 1, "DetectionHead", [256]],                  
            [18, 1, "DetectionHead", [512]],                  
            [21, 1, "DetectionHead", [512]],                  
            [[22, 23, 24], 1, "Detect", [nc]],
        ],
    }
    p = Path(save_dir) / f"{tag}.yaml"
    with open(p,"w") as f: yaml.dump(d, f, default_flow_style=False, sort_keys=False)
    return p




def main():
    total_start = time.time()
    cfg = Config()
    cfg.workers = safe_worker_count(cfg.workers)
    data_yaml    = Path(cfg.data_root) / cfg.dataset / "data.yaml"
    results_base = Path(cfg.results_root)
    results_base.mkdir(parents=True, exist_ok=True)
    if not data_yaml.exists(): raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    with open(data_yaml) as f: nc = yaml.safe_load(f).get("nc", 12)

    print("="*65)
    print(f"  EXP #{EXP_ID} — {EXP_NAME.upper()}")
    print("  Backbone: YOLOv8l (standard)   Neck: PAN")
    print("  Head: DetectionHead + Detect")
    print(f"  Epochs: {cfg.epochs}  Batch: {cfg.batch_size}  Device: {cfg.device}")
    print("="*65)

    register_custom_modules()
    yaml_path = build_yaml(nc, EXP_NAME, results_base)

    job_id   = os.environ.get("SLURM_JOB_ID", f"local_{int(time.time())}")
    run_name = f"exp{EXP_ID}_{EXP_NAME}_{job_id}"
    tracker  = MetricsTracker(results_base, cfg.dataset, run_name, total_start)
    cfg.save(tracker.run_dir / "config.yaml")

    model = YOLO(str(yaml_path))
    for p in model.model.parameters(): p.data = p.data.clone()
    model.add_callback("on_before_zero_grad", make_loss_patch_cb(cfg.fl_gamma, cfg.label_smoothing))
    model.add_callback("on_train_epoch_start", make_copy_paste_cb(cfg.copy_paste_start_epoch, cfg.copy_paste))

    model.train(
        data=str(data_yaml), epochs=cfg.epochs, imgsz=cfg.image_size,
        batch=cfg.batch_size, device=cfg.device, workers=cfg.workers,
        project=str(results_base/cfg.dataset), name=run_name,
        exist_ok=True, save=True, save_period=0, verbose=cfg.verbose, plots=cfg.plots, freeze=0,
        optimizer="AdamW", lr0=cfg.lr0, lrf=cfg.lrf, weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs, warmup_momentum=cfg.warmup_momentum,
        warmup_bias_lr=cfg.warmup_bias_lr,
        fliplr=cfg.fliplr, translate=cfg.translate, scale=cfg.scale,
        mosaic=cfg.mosaic, close_mosaic=cfg.close_mosaic,
        copy_paste=0.0, mixup=cfg.mixup,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0,
        patience=cfg.patience, cos_lr=True, seed=_SEED,
    )

    best = results_base/cfg.dataset/run_name/"weights"/"best.pt"
    register_custom_modules()
    bm = YOLO(str(best))
    bm.val(data=str(data_yaml), imgsz=cfg.image_size, batch=cfg.batch_size, device=cfg.device, split="val")
    test_dir = Path(cfg.data_root)/cfg.dataset/"images"/"test"
    if test_dir.exists():
        res = bm.val(data=str(data_yaml), split="test", imgsz=cfg.image_size,
                     batch=cfg.batch_size, device=cfg.device, augment=cfg.use_tta)
        tracker.update(res)
    print(f"\n   Exp #{EXP_ID} done in {(time.time()-total_start)/3600:.2f} h     {tracker.run_dir}")

if __name__ == "__main__":
    main()
