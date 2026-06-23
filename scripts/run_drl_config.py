"""
Standalone CLI entrypoint: run ONE DRL config end-to-end (warm-start + training).

Purpose: let two independent configs train concurrently on separate GPUs from one
Kaggle "GPU T4 x2" session, via subprocess + CUDA_VISIBLE_DEVICES. A real OS
subprocess (not multiprocessing.Process) is used deliberately:
  - CUDA_VISIBLE_DEVICES must be set BEFORE the first `import torch` in a process;
    a subprocess gets a clean environment, a Process forked/spawned from a notebook
    kernel may not.
  - Jupyter's 'spawn' start method requires worker targets to be importable by
    reference; a function defined inline in a notebook cell is not. A subprocess
    sidesteps this entirely — it just runs this file.

Usage (one call per GPU, launched concurrently):
    CUDA_VISIBLE_DEVICES=0 python scripts/run_drl_config.py \\
        --pkg-root /kaggle/input/.../iteris-pkg \\
        --drl-config configs/camus_drl_c1.yaml --agent TD3 \\
        --data-root /kaggle/input/.../CAMUS \\
        --checkpoint /kaggle/input/.../camus_lite_unet_best.pt \\
        --checkpoint-dir /kaggle/working/c1_td3

See the bottom of this file / the project notebooks for the subprocess.Popen
pattern that launches two of these concurrently with CUDA_VISIBLE_DEVICES=0/1.
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pkg-root', required=True, help='directory containing iteris/ and configs/')
    p.add_argument('--drl-config', required=True,
                   help='path to a configs/*_drl_*.yaml, absolute or relative to --pkg-root')
    p.add_argument('--agent', required=True, help="AGENT_NAME, e.g. 'TD3' or 'DuelingDDQN'")
    p.add_argument('--data-root', required=True,
                   help='dataset root (CAMUS folder, or BRISC segmentation_task folder)')
    p.add_argument('--checkpoint', required=True, help='baseline (lite/attn) U-Net checkpoint path')
    p.add_argument('--checkpoint-dir', required=True, help='output dir for this run')
    p.add_argument('--tumor-type-filter', default=None, help='BRISC only: glioma|meningioma|pituitary')
    args = p.parse_args()

    sys.path.insert(0, args.pkg_root)
    from iteris.config import load_drl_class_config, resolve_agent_config, load_config
    from iteris.warm_start import precompute_init_masks
    from iteris.drl_training import run_drl_training
    from iteris.utils import get_device

    pkg_root = Path(args.pkg_root)
    drl_cfg_path = (args.drl_config if Path(args.drl_config).is_absolute()
                    else str(pkg_root / args.drl_config))

    cfg_full = load_drl_class_config(drl_cfg_path)
    cfg = resolve_agent_config(cfg_full, args.agent)
    baseline_cfg = load_config(str(pkg_root / 'configs' / cfg['baseline_cfg_name']))
    baseline_cfg['data_root'] = args.data_root
    cfg['baseline_checkpoint'] = args.checkpoint
    cfg['checkpoint_dir'] = args.checkpoint_dir
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if args.tumor_type_filter:
        cfg['tumor_type_filter'] = args.tumor_type_filter

    print(f'[run_drl_config] device={get_device()} agent={cfg["agent_type"]} '
          f'class={cfg.get("class_name")} dataset={cfg["dataset"]} '
          f'checkpoint_dir={args.checkpoint_dir}', flush=True)

    train_samples, val_samples, _test_samples = precompute_init_masks(
        baseline_cfg=baseline_cfg,
        baseline_checkpoint=cfg['baseline_checkpoint'],
        target_class=cfg['target_class'],
        min_area_fraction=cfg.get('min_area_fraction', 0.01),
        tumor_type_filter=cfg.get('tumor_type_filter'),
    )

    result = run_drl_training(cfg, train_samples, val_samples)

    summary = {
        'agent_type': cfg['agent_type'],
        'class_name': cfg.get('class_name'),
        'dataset':    cfg['dataset'],
        'best_dice':  result['best_dice'],
        'checkpoint': result['checkpoint'],
    }
    out_path = Path(args.checkpoint_dir) / 'run_summary.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(f'[run_drl_config] DONE best_dice={result["best_dice"]:.4f} -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
