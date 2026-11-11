
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import argparse
import csv
import json
import pprint
import sys
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass, asdict
import yaml
import torch

EXP_ID = 'v6l'
EXP_NAME = 'yolov6l_native_std'

@dataclass
class Config:
    data: str = '/home/ramyads/YOLO/Datasets/o1/data.yaml'
    results_root: str = '/home/ramyads/YOLO/results'
    epochs: int = 200
    batch: int = 8
    imgsz: int = 640
    device: str = '0'
    workers: int = 4
    weights: str = 'weights/yolov6l.pt'
    scratch: bool = False
    fuse_ab: bool = False
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
    weights = Path(options.weights).expanduser()
    weights = weights if weights.is_absolute() else root / weights
    cfg.model.pretrained = None if options.scratch else str(weights.resolve())
    cfg.solver.optim = 'AdamW'
    cfg.solver.lr0 = options.lr0
    cfg.solver.lrf = options.lrf
    cfg.solver.weight_decay = options.weight_decay
    cfg.solver.momentum = .9
    cfg.solver.warmup_epochs = options.warmup_epochs
    cfg.solver.warmup_momentum = options.warmup_momentum
    cfg.solver.warmup_bias_lr = options.warmup_bias_lr
    cfg.data_aug.update(dict(hsv_h=0., hsv_s=0., hsv_v=0., degrees=0.,
                             translate=.1, scale=.5, shear=0., flipud=0.,
                             fliplr=.5, mosaic=1., mixup=.1))
    cfg.eval_params = dict(conf_thres=.001, iou_thres=.6, do_coco_metric=False,
                           do_pr_metric=True, plot_curve=True, plot_confusion_matrix=False)
    return cfg


def build_adamw(cfg, model):
    """Include every trainable parameter exactly once.

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
    if options.epochs < 1 or options.batch < 1 or options.imgsz % 32:
        raise ValueError('Use positive epochs/batch and an image size divisible by 32.')
    source, data = resolve_data(options.data)
    project = Path(options.results_root).expanduser().resolve() / source.parent.name
    run = 'exp' + EXP_ID + '_' + EXP_NAME + '_' + time.strftime('%Y%m%d_%H%M%S')
    run_dir = project / run
    run_dir.mkdir(parents=True, exist_ok=False)
    data_path = run_dir / 'data_absolute.yaml'
    data_path.write_text(yaml.safe_dump(data))
    cfg = architecture(root, options)
    if cfg.model.pretrained:
        weights_path = Path(cfg.model.pretrained)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        if not weights_path.exists() and weights_path.name != 'yolov6l.pt':
            raise FileNotFoundError(f'Explicit checkpoint does not exist: {weights_path}')
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
                          bs_per_gpu=options.batch, fuse_ab=options.fuse_ab, distill=False).items():
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
    # Keep the original five-column aggregate schema, test results only.
    if data.get('test'):
        aggregate = Path(options.results_root).expanduser().resolve() / 'all_models_metrics.csv'
        fields = ['exp_id', 'exp_name', 'dataset', 'map50', 'map50_95']
        if aggregate.exists():
            with aggregate.open(newline='') as f:
                if next(csv.reader(f), None) != fields:
                    raise ValueError(f'Unexpected CSV header in {aggregate}; per-run evaluation.csv is already saved.')
        new = not aggregate.exists()
        with aggregate.open('a', newline='') as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(fields)
            writer.writerow([EXP_ID, EXP_NAME, source.parent.name, float(result[0]), float(result[1])])
    print(f'Done: {best}')



def smoke_test(root):
    """Random weights and synthetic labels: no checkpoint download or dataset."""
    from yolov6.models.yolo import build_model
    from yolov6.models.losses.loss import ComputeLoss
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(42)
    cfg = architecture(root, Config(scratch=True))
    model = build_model(cfg, 3, torch.device('cpu'), fuse_ab=False)
    assert type(model.backbone).__name__ == 'CSPBepBackbone'
    assert type(model.neck).__name__ == 'CSPRepBiFPANNeck'
    assert model.stride.tolist() == [8, 16, 32]
    model.eval()
    with torch.no_grad():
        for h, w in ((128, 128), (128, 192)):
            prediction, features = model(torch.zeros(1, 3, h, w))
            assert prediction.shape == (1, sum((h//s)*(w//s) for s in (8, 16, 32)), 8)
            assert [tuple(f.shape[-2:]) for f in features] == [(h//s, w//s) for s in (8, 16, 32)]
    model.train()
    outputs, _ = model(torch.randn(2, 3, 128, 128))
    criterion = ComputeLoss(num_classes=3, ori_img_size=128, warmup_epoch=0, use_dfl=True, reg_max=16)
    targets = torch.tensor([[0, 1, .5, .5, .2, .2], [1, 2, .3, .4, .1, .15]])
    loss, parts = criterion(outputs, targets, epoch_num=1, step_num=0)
    assert torch.isfinite(loss).all() and torch.isfinite(parts).all()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detect.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer = build_adamw(cfg, model)  # Checks complete parameter coverage.
    del optimizer, outputs, loss
    model.zero_grad(set_to_none=True)
    model.eval()
    model.nc, model.names = 3, ['a', 'b', 'c']
    with torch.no_grad(), tempfile.TemporaryDirectory() as td:
        x = torch.zeros(1, 3, 128, 128)
        expected = model(x)[0]
        ckpt = Path(td) / 'test.pt'
        torch.save({'model': model, 'ema': None}, ckpt)
        from yolov6.utils.checkpoint import load_checkpoint
        restored = load_checkpoint(str(ckpt), map_location='cpu', fuse=False)
        torch.testing.assert_close(expected, restored(x)[0])
    print('PASS: stock architecture, output shapes, native loss/backward, optimizer coverage, checkpoint reload.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--data', default=Config.data)
    parser.add_argument('--results-root', default=Config.results_root)
    parser.add_argument('--device', default=Config.device)
    parser.add_argument('--epochs', type=int, default=Config.epochs)
    parser.add_argument('--batch-size', type=int, default=Config.batch)
    parser.add_argument('--workers', type=int, default=Config.workers)
    parser.add_argument('--weights', default=Config.weights)
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--fuse-ab', action='store_true', help='Enable native anchor-aided auxiliary training.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if not (root / 'yolov6/models/yolo.py').is_file():
        raise SystemExit('Save this file in the root of a meituan/YOLOv6 tag 0.4.0 checkout.')
    os.chdir(root)
    if args.smoke_test:
        smoke_test(root)
    else:
        # SLURM_CPUS_ON_NODE may contain a suffix such as 8(x2).
        import re
        raw = os.getenv('SLURM_CPUS_ON_NODE', '')
        matched = re.match(r'\d+', raw)
        available = int(matched.group()) if matched else (os.cpu_count() or 4)
        workers = min(max(0, args.workers), max(1, available // 2))
        train_model(root, Config(data=args.data, results_root=args.results_root,
                                epochs=args.epochs, batch=args.batch_size, device=args.device,
                                workers=workers, weights=args.weights, scratch=args.scratch,
                                fuse_ab=args.fuse_ab))


if __name__ == '__main__':
    main()
