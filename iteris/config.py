"""Load and validate YAML configs."""

from pathlib import Path
from typing import Union
import yaml


# Minimal required keys — anything missing raises early.
REQUIRED_KEYS = {
    'dataset', 'modality', 'data_root', 'checkpoint_dir',
    'image_size', 'num_classes', 'class_names', 'class_colors',
    'normalize', 'val_split', 'test_split', 'label_frac',
    'batch_size', 'epochs', 'lr', 'weight_decay', 'patience', 'seed',
}

# Valid normalisation modes — must match transforms.build_intensity_transform.
VALID_NORMALIZE = {'minmax', 'zscore', 'hu'}


def load_config(path: Union[str, Path]) -> dict:
    """
    Load a YAML config and validate the required keys.

    Override any field from the notebook after loading by mutating the dict:
        cfg = load_config('configs/camus.yaml')
        cfg['data_root'] = '/some/other/path'

    Raises
    ------
    FileNotFoundError  if path does not exist
    KeyError           if a required key is missing
    ValueError         if normalize is invalid or HU window is missing for ct
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Config not found: {path}')

    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    missing = REQUIRED_KEYS - set(cfg)
    if missing:
        raise KeyError(f'Config {path} is missing keys: {sorted(missing)}')

    if cfg['normalize'] not in VALID_NORMALIZE:
        raise ValueError(
            f"normalize='{cfg['normalize']}' not in {VALID_NORMALIZE}"
        )

    if cfg['normalize'] == 'hu' and 'hu_window' not in cfg:
        raise ValueError("normalize='hu' requires 'hu_window: [a_min, a_max]' in config")

    # Tuple-ify list-typed numeric fields for downstream type consistency
    cfg['image_size'] = tuple(cfg['image_size'])
    if 'spacing' in cfg:
        cfg['spacing'] = tuple(cfg['spacing'])

    return cfg


def load_drl_config(path: Union[str, Path]) -> dict:
    """
    Load a DRL agent config (skips baseline-specific validation).

    DRL configs only need a handful of keys: agent_type, target_class,
    train_steps, etc. They share the iteris infrastructure but train
    different objects (agents, not U-Nets).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'DRL config not found: {path}')
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    if 'agent_type' not in cfg:
        raise KeyError(f'DRL config missing required key: agent_type')
    return cfg
