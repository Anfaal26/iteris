"""Load and validate YAML configs."""

from pathlib import Path
from typing import Dict, Union
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
        cfg = load_config('configs/CAMUS/camus.yaml')
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


def load_drl_class_config(path: Union[str, Path]) -> dict:
    """
    Load a per-class DRL config (camus_drl_c1.yaml, etc.).

    Per-class configs contain:
      - class-specific env / reward params at the top level
        (target_class, reward_mode, hd_norm, shift_px, …)
      - an ``agents:`` block with per-algorithm hyperparams

    They do NOT have ``agent_type`` at the top level — that lives inside
    each agents sub-block.  Use ``resolve_agent_config()`` after loading
    to merge the chosen agent's hyperparams into a flat training dict.

    Raises
    ------
    FileNotFoundError  if path does not exist
    KeyError           if ``target_class`` or ``agents`` is missing
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'DRL class config not found: {path}')
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    for key in ('target_class', 'agents'):
        if key not in cfg:
            raise KeyError(f'DRL class config {path} is missing required key: {key}')
    return cfg


def resolve_agent_config(cfg: dict, agent_name: str) -> dict:
    """
    Merge per-agent hyperparams from a per-class config into a flat dict.

    Per-class configs (loaded via ``load_drl_class_config``) have the shape::

        target_class: 1
        reward_mode:  dice_hd_composite
        ...                              # env / reward params (shared)
        agents:
          DQN:
            lr: 1.0e-4
            ...
          DuelingDDQN:
            lr: 1.0e-4
            ...
          DDPG:
            actor_lr: 1.0e-4
            ...

    This function extracts ``cfg['agents'][agent_name]`` and merges it over
    the top-level dict (agent params win on conflict), then sets
    ``cfg['agent_type']`` so the training loop can pick the right class.

    Parameters
    ----------
    cfg        : dict returned by ``load_drl_class_config``
    agent_name : selector key matching a block under ``cfg['agents']``,
                 e.g. 'DQN' | 'DuelingDDQN' | 'DDPG' | 'DQN_TRACE' | 'DuelingDDQN_TRACE'
                 (matched case-insensitively).

    Returns
    -------
    Flat dict ready to pass directly to ``run_drl_training``.

    Raises
    ------
    KeyError  if agent_name is not present in cfg['agents']
    """
    agents   = cfg.get('agents', {})
    available = list(agents.keys())
    # Match the selector key case-insensitively so 'DuelingDDQN' resolves
    # whether the user types 'DuelingDDQN', 'duelingddqn', or 'DUELINGDDQN'.
    if agent_name in agents:
        agent_key = agent_name
    else:
        lowered = {k.lower(): k for k in agents}
        if agent_name.lower() in lowered:
            agent_key = lowered[agent_name.lower()]
        else:
            raise KeyError(
                f"Agent '{agent_name}' not found in config. "
                f"Available: {available}"
            )
    # Start from top-level (env / reward / class params), strip 'agents' block
    merged = {k: v for k, v in cfg.items() if k != 'agents'}
    # Layer agent-specific hyperparams on top (override where keys conflict)
    merged.update(agents[agent_key])
    # Ensure agent_type is set. Block-defined agent_type wins so a selector
    # key can differ from the underlying algorithm — e.g. the 'DuelingDDQN'
    # block sets ``agent_type: DUELING`` (which maps to DuelingDQNAgent in
    # AGENT_REGISTRY) while keeping a friendlier selector name in configs.
    merged.setdefault('agent_type', agent_key)
    return merged


# ── Contour-refinement config (single source of truth) ──────────────────────
# Replaces the per-notebook ``cfg.update({...})`` blocks and the scattered
# ``if AGENT_NAME != 'TD3': cfg['max_steps'] = 50`` conditionals with ONE
# agent-aware call, made right after ``resolve_agent_config``. The 8 DRL
# notebooks stay byte-identical except for the two things that genuinely
# differ per run: ``baseline_cfg_name`` (which U-Net to refine) and
# ``uncertainty_gate`` (ON only where the diagnostic found a USABLE prob_map).

_REFINE_SHARED: Dict = dict(
    # Reward -- dense per-control-point distance-to-GT-boundary (not global
    # Dice): self-regularising, so the agent stops instead of overshooting.
    reward_mode='contour_boundary',
    reward_clip=10.0,              # deltas are in pixels, not Dice units
    reward_step_penalty=0.0,       # DuelingDDQN's YAML 0.05 drove an 88%-STOP collapse
    # Optimal-stopping STOP incentive (discrete agents only -- continuous TD3 has
    # no STOP action). A CHOSEN STOP earns terminal_bonus_scale*(dice - dice_0);
    # timing out at max_steps earns nothing (see ContourRefineEnv.step /
    # _terminal_step). The dense reward alone left STOP unlearnable near the peak
    # (~0/noisy margin -> 12% / 0% STOP observed, good masks then edited past their
    # peak); this gives a clean, above-noise target for WHEN to commit and also
    # sharpens the value fn (Q peaks at high-dice states -> better value-floored
    # deploy). Scale 20: a +0.05-Dice gain -> +1.0 bonus (~a few dense steps),
    # a near-converged +0.005 -> +0.1 (correctly small). PROVISIONAL -- validate
    # via the 10/20/40 sweep in the trial notebook before trusting the magnitude.
    terminal_bonus_scale=20.0,

    # GT-free routing gate -- train/eval only on cases whose init mask is a
    # single dominant, plausibly-sized component; excluded cases are ROUTED to
    # the raw U-Net mask at eval (never GT-dropped).
    refinable_gate=True,
    refinable_min_cc_frac=0.004,
    refinable_min_dominance=0.5,

    # Fine-step mechanics -- step size finer than the residual boundary error
    # (~2.5px) so pushes converge instead of overshoot.
    disp_px=0.5,
    auto_smooth_lambda=0.1,        # per-step Laplacian smoothing (continuous has no SMOOTH action)
    curriculum_max_steps=False,    # every sample gets the full step budget
    disable_auto_stop=True,        # env's stop_eps_dice was tuned for disp_px=2.0

    # Perception -- per-sector LOCAL features (DeepSnake/MARL), not one pooled
    # vector deciding every sector.
    spatial_head=True,

    # BC warm-start -- GT-oracle imitation for BOTH agents (fair comparison).
    bc_warm_start=True,
    bc_demo_episodes=80,

    # Ablations -- unproven / not applicable here.
    directional_state=False,
    topology_filter=False,
)

# The continuous actor moves EVERY sector each step; the discrete agent moves
# ONE sector/step, so it needs a higher step CAP and deeper BC demos to reach /
# demonstrate a distributed correction. Discrete self-terminates via its learned
# STOP, so max_steps is a ceiling, not a fixed length (research: single-element-
# per-step RL refinement uses tens-to-hundreds of steps/episode).
_REFINE_CONTINUOUS: Dict = dict(max_steps=25, bc_demo_max_steps=12)
_REFINE_DISCRETE:   Dict = dict(max_steps=50, bc_demo_max_steps=30)

# agent_type (from resolve_agent_config) -> which override set.
CONTINUOUS_AGENTS = {'TD3', 'DDPG'}
DISCRETE_AGENTS = {'DUELING', 'DQN'}


def is_continuous_agent(agent_type: str) -> bool:
    """True for TD3/DDPG (continuous action), False for DUELING/DQN (discrete)."""
    return str(agent_type).upper() in CONTINUOUS_AGENTS


def apply_refinement_config(cfg: dict, *, baseline_cfg_name: str,
                            uncertainty_gate: bool = False,
                            verbose: bool = True) -> dict:
    """Apply the full contour-refinement config to a resolved agent ``cfg``,
    IN PLACE, choosing the TD3-vs-discrete specific settings from
    ``cfg['agent_type']`` (set by ``resolve_agent_config``). Returns ``cfg``.

    Call this right after ``resolve_agent_config``:

        cfg = resolve_agent_config(cfg_full, AGENT_NAME)
        apply_refinement_config(cfg, baseline_cfg_name='CAMUS/camus.yaml',
                                uncertainty_gate=False)

    Args:
        cfg: the per-agent cfg from ``resolve_agent_config``.
        baseline_cfg_name: e.g. 'CAMUS/camus.yaml' / 'BRISC/brisc.yaml'
            (the ATTENTION U-Net; Phase B adds label_frac downstream).
        uncertainty_gate: True only where the prob_map diagnostic is USABLE.
        verbose: print a one-line summary (goes to the run log).
    """
    at = str(cfg.get('agent_type', '')).upper()
    if at not in CONTINUOUS_AGENTS and at not in DISCRETE_AGENTS:
        raise ValueError(
            f"apply_refinement_config: unrecognised agent_type {at!r} "
            f"(expected one of {CONTINUOUS_AGENTS | DISCRETE_AGENTS}). "
            f"Was resolve_agent_config() called first?")

    cfg['baseline_cfg_name'] = baseline_cfg_name
    cfg.update(_REFINE_SHARED)
    cfg['uncertainty_gate'] = bool(uncertainty_gate)
    cfg.update(_REFINE_CONTINUOUS if at in CONTINUOUS_AGENTS else _REFINE_DISCRETE)

    if verbose:
        kind = 'continuous' if at in CONTINUOUS_AGENTS else 'discrete'
        print(f"[config] agent={at} ({kind}) base={baseline_cfg_name} "
              f"reward={cfg['reward_mode']} uncertainty_gate={cfg['uncertainty_gate']} "
              f"refinable_gate={cfg['refinable_gate']} disp_px={cfg['disp_px']} "
              f"max_steps={cfg['max_steps']} spatial_head={cfg['spatial_head']} "
              f"bc_warm_start={cfg['bc_warm_start']} bc_demo_max_steps={cfg['bc_demo_max_steps']}")
    return cfg
