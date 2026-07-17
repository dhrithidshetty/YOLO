

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["NCCL_P2P_DISABLE"]               = "1"
os.environ["NCCL_SHM_DISABLE"]               = "1"
os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"]= "1"
os.environ["CUDA_DEVICE_ORDER"]              = "PCI_BUS_ID"

import random, time, csv, yaml, builtins
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


from dataclasses import dataclass, asdict

@dataclass
class Config:
    dataset      : str   = "o1"
    data_root    : str   = "/home/ramyads/YOLO/Datasets"
    results_root : str   = "/home/ramyads/YOLO/results"
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
    def save(self, p):
        with open(p,"w") as f: yaml.dump(asdict(self),f,default_flow_style=False)

def safe_worker_count(req):
    slurm = int(os.environ.get("SLURM_CPUS_ON_NODE",0))
    avail = slurm if slurm>0 else (os.cpu_count() or 4)
    cap   = max(1, avail//2)
    n     = min(req, cap)
    if n < req: print(f"  \u2139  workers capped {req} \u2192 {n}")
    return n

EFFNET_SPECS = {"b2": {"p3":(3,48),"p4":(5,120),"p5":(7,352)}}

class SEBlock(nn.Module):
    def __init__(self,ch,r=4):
        super().__init__()
        rc=max(1,ch//r)
        self.fc=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(ch,rc,1),
                               nn.SiLU(inplace=True),nn.Conv2d(rc,ch,1),nn.Sigmoid())
    def forward(self,x): return x*self.fc(x)

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

class _ExtractScale(nn.Module):
    def __init__(self,oc,idx):
        super().__init__()
        self.idx=idx; self.out_channels=oc
    def forward(self,x): return x[self.idx] if isinstance(x,(list,tuple)) else x


class EffNetFullBiFPNDH(nn.Module):
    out_channels: List[int]
    def __init__(self, model_size='b2', pretrained=True, bifpn_channels=192, cbam_reduction=16, se_reduction=4, bifpn_rounds=2, **_):
        super().__init__()
        ch=bifpn_channels
        w=getattr(models,f'EfficientNet_{model_size.upper()}_Weights').DEFAULT if pretrained else None
        ef=getattr(models,f'efficientnet_{model_size}')(weights=w)
        self.features=nn.Sequential(*list(ef.features.children()))
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


class _FocalBCE(nn.Module):
    def __init__(self,b,g=1.5):
        super().__init__()
        self.g=g; self.r=b.reduction; self.pw=b.pos_weight
        self._b=nn.BCEWithLogitsLoss(reduction="none",pos_weight=b.pos_weight)
    def forward(self,p,t):
        L=self._b(p,t); pt=torch.sigmoid(p)*t+(1-torch.sigmoid(p))*(1-t)
        L=(1-pt)**self.g*L
        return L.mean() if self.r=="mean" else (L.sum() if self.r=="sum" else L)

def _gc(tr): return getattr(tr,"criterion",None) or getattr(tr,"compute_loss",None)

def _pf(tr,g):
    if g<=0: return
    c=_gc(tr)
    for a in ("bce","bce_cls","bce_obj","cls_loss_fn"):
        n=getattr(c,a,None)
        if isinstance(n,nn.BCEWithLogitsLoss):
            setattr(c,a,_FocalBCE(n,g)); print(f"    fl_gamma={g} on criterion.{a}"); return
    for cn,ch in c.named_children():
        for a in ("bce","bce_cls","bce_obj","cls_loss_fn"):
            n=getattr(ch,a,None)
            if isinstance(n,nn.BCEWithLogitsLoss):
                setattr(ch,a,_FocalBCE(n,g)); print(f"    fl_gamma={g} on criterion.{cn}.{a}"); return
    print("    fl_gamma: no BCEWithLogitsLoss found")

def _pl(tr,ls):
    if ls<=0: return
    c=_gc(tr)
    if hasattr(c,"label_smoothing"): c.label_smoothing=ls; print(f"    label_smoothing={ls}"); return
    for cn,ch in c.named_children():
        if hasattr(ch,"label_smoothing"):
            ch.label_smoothing=ls; print(f"   label_smoothing={ls} on .{cn}"); return
    try: c.label_smoothing=ls; print(f"    label_smoothing={ls} force-set")
    except Exception as e: print(f"   label_smoothing failed: {e}")

def make_loss_cb(fg,ls):
    done={"v":False}
    def cb(tr):
        if done["v"]: return
        print("\n\U0001f527 Loss patches batch-0")
        _pf(tr,fg); _pl(tr,ls); done["v"]=True
    return cb

def make_cp_cb(ep,v):
    def cb(tr):
        if tr.epoch==ep:
            tr.args.copy_paste=v
            print(f"\n    copy_paste={v} at epoch {ep}")
    return cb

class MetricsTracker:
    def __init__(self,base,ds,run,eid,ename,t0=None):
        self.base=Path(base); self.ds=ds; self.run=run; self.eid=eid; self.ename=ename
        self.t0=t0 or time.time()
        self.csv=self.base/"all_models_metrics.csv"
        self.rd=self.base/ds/run; self.rd.mkdir(parents=True,exist_ok=True)
        if not self.csv.exists():
            with open(self.csv,"w",newline="") as f:
                csv.writer(f).writerow(["exp_id","exp_name","dataset","map50","map50_95"])
    def update(self,r):
        m50=m5095=0.0
        if hasattr(r,"box"):
            m50=float(getattr(r.box,"map50",0.0)); m5095=float(getattr(r.box,"map",0.0))
        with open(self.csv,"a",newline="") as f:
            csv.writer(f).writerow([self.eid,self.ename,self.ds,m50,m5095])
        print(f"    mAP50={m50:.4f}  mAP50-95={m5095:.4f}")

def register():
    import ultralytics.nn.tasks as T
    import ultralytics.nn.modules as M
    for c in (SEBlock, CBAMBlock, ConvBNSiLU, _CBN, SPPF_, BiFPNConv, BiFPNLayer, BiFPN, DetectionHead, _ExtractScale, EffNetFullBiFPNDH):
        n=c.__name__
        setattr(M,n,c); M.__dict__[n]=c
        setattr(T,n,c); T.__dict__[n]=c
        setattr(builtins,n,c)
    print(f"    Registered: {[c.__name__ for c in (SEBlock, CBAMBlock, ConvBNSiLU, _CBN, SPPF_, BiFPNConv, BiFPNLayer, BiFPN, DetectionHead, _ExtractScale, EffNetFullBiFPNDH)]}")

def main():
    t0=time.time()
    cfg=Config()
    cfg.workers=safe_worker_count(cfg.workers)
    dy=Path(cfg.data_root)/cfg.dataset/"data.yaml"
    rb=Path(cfg.results_root); rb.mkdir(parents=True,exist_ok=True)
    if not dy.exists(): raise FileNotFoundError(f"data.yaml not found: {dy}")
    with open(dy) as f: nc=yaml.safe_load(f).get("nc",12)
    ch=cfg.bifpn_channels

    print("="*65)
    print(f"  EXP #12 — EFFNETB2_FULLSTACK_BIFPN_DETHEAD")
    print("  *** FULL PROPOSED MODEL ***")
    print("  Backbone: EffNet-B2+Proj+CBAM+SPPF+BiFPN+DetHead (all inside) [FIXED]")
    print("  Head: Detect")
    print(f"  Epochs:{cfg.epochs}  Batch:{cfg.batch_size}  Device:{cfg.device}")
    print("="*65)

    register()
    yp=rb/f"_EffNetFullBiFPNDH_bifpn_dh_{nc}.yaml"
    yaml.dump({"nc":nc,"backbone":[
        [-1,1,"EffNetFullBiFPNDH",["b2",True,ch,cfg.cbam_reduction,cfg.se_reduction,cfg.bifpn_rounds]],
        [0,1,"_ExtractScale",[ch,0]],
        [0,1,"_ExtractScale",[ch,1]],
        [0,1,"_ExtractScale",[ch,2]],
    ],"head":[
        [[1,2,3],1,"Detect",[nc]],
    ]},open(yp,"w"),default_flow_style=False,sort_keys=False)

    jid=os.environ.get("SLURM_JOB_ID",f"local_{int(time.time())}")
    rn=f"exp12_effnetb2_fullstack_bifpn_dethead_{jid}"
    tr=MetricsTracker(rb,cfg.dataset,rn,"12","effnetb2_fullstack_bifpn_dethead",t0)
    cfg.save(tr.rd/"config.yaml")

    model=YOLO(str(yp))
    for p in model.model.parameters(): p.data=p.data.clone()
    model.add_callback("on_before_zero_grad",make_loss_cb(cfg.fl_gamma,cfg.label_smoothing))
    model.add_callback("on_train_epoch_start",make_cp_cb(cfg.copy_paste_start_epoch,cfg.copy_paste))
    TRAIN_KW=dict(
        data=str(dy),epochs=cfg.epochs,imgsz=cfg.image_size,
        batch=cfg.batch_size,device=cfg.device,workers=cfg.workers,
        project=str(rb/cfg.dataset),name=rn,
        exist_ok=True,save=True,save_period=0,verbose=cfg.verbose,plots=cfg.plots,freeze=0,
        optimizer="AdamW",lr0=cfg.lr0,lrf=cfg.lrf,weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,warmup_momentum=cfg.warmup_momentum,
        warmup_bias_lr=cfg.warmup_bias_lr,
        fliplr=cfg.fliplr,translate=cfg.translate,scale=cfg.scale,
        mosaic=cfg.mosaic,close_mosaic=cfg.close_mosaic,
        copy_paste=0.0,mixup=cfg.mixup,
        hsv_h=0.0,hsv_s=0.0,hsv_v=0.0,degrees=0.0,shear=0.0,perspective=0.0,flipud=0.0,
        patience=cfg.patience,cos_lr=True,seed=_SEED)

    model.train(**TRAIN_KW)

    best=rb/cfg.dataset/rn/"weights"/"best.pt"
    register()
    bm=YOLO(str(best))
    bm.val(data=str(dy),imgsz=cfg.image_size,batch=cfg.batch_size,device=cfg.device,split="val")
    td=Path(cfg.data_root)/cfg.dataset/"images"/"test"
    if td.exists():
        r=bm.val(data=str(dy),split="test",imgsz=cfg.image_size,
                 batch=cfg.batch_size,device=cfg.device,augment=cfg.use_tta)
        tr.update(r)
    print(f"\n  Exp #12 done in {(time.time()-t0)/3600:.2f} h  \u2192  {tr.rd}")

if __name__=="__main__":
    main()
