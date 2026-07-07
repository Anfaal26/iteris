"""Single source of truth for the Iteris contour-refinement DRL config.

Replaces the per-notebook ``cfg.update({...})`` blocks and the scattered
``if AGENT_NAME != 'TD3': cfg['max_steps'] = 50`` conditionals with ONE
agent-aware call. The 8 DRL notebooks then stay byte-identical except for the
two things that genuinely differ per run:

    * ``baseline_cfg_name`` -- which U-Net to refine (CAMUS vs BRISC).
    * ``uncertainty_gate``  -- only ON where the diagnostic found a USABLE
      (graded) prob_map; OFF (default) where it is INERT/near-binary.

All the reasoning that used to live in the notebook comments now lives here
(and in docs/EXPERIMENTS.md), version-controlled and testable, so a fix lands
in ONE place instead of eight notebooks.

Usage (in the Configure cell, right after resolve_agent_config):

    from iteris.refine_config import apply_refinement_config
    cfg = resolve_agent_config(cfg_full, AGENT_NAME)
    apply_refinement_config(cfg, baseline_cfg_name='CAMUS/camus.yaml',
                            uncertainty_gate=False)
"""
from typing import Dict

# ── Config shared by BOTH agents ─────────────────────────────────────────────
# Refine the ATTENTION U-Net (boundary-error regime a contour agent can fix,
# not the lite net's structural errors) with a dense per-control-point
# boundary reward, a GT-free routing gate, fine-step mechanics, per-sector
# local perception, and a shared GT-oracle BC warm-start.
_SHARED: Dict = dict(
    # Reward -- dense per-control-point distance-to-GT-boundary (not global
    # Dice): self-regularising, so the agent stops instead of overshooting.
    reward_mode='contour_boundary',
    reward_clip=10.0,              # deltas are in pixels, not Dice units
    reward_step_penalty=0.0,       # DuelingDDQN's YAML 0.05 drove an 88%-STOP collapse

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

# ── Per-agent overrides ──────────────────────────────────────────────────────
# The continuous actor moves EVERY sector each step; the discrete agent moves
# ONE sector/step, so it needs a higher step CAP and deeper BC demos to reach /
# demonstrate a distributed correction. Discrete self-terminates via its learned
# STOP, so max_steps is a ceiling, not a fixed length (research: single-element-
# per-step RL refinement uses tens-to-hundreds of steps/episode).
_CONTINUOUS: Dict = dict(max_steps=25, bc_demo_max_steps=12)
_DISCRETE:   Dict = dict(max_steps=50, bc_demo_max_steps=30)

# agent_type (from resolve_agent_config) -> which override set.
_CONTINUOUS_AGENTS = {'TD3', 'DDPG'}
_DISCRETE_AGENTS = {'DUELING', 'DQN'}


def is_continuous_agent(agent_type: str) -> bool:
    """True for TD3/DDPG (continuous action), False for DUELING/DQN (discrete)."""
    return str(agent_type).upper() in _CONTINUOUS_AGENTS


def apply_refinement_config(cfg: dict, *, baseline_cfg_name: str,
                            uncertainty_gate: bool = False,
                            verbose: bool = True) -> dict:
    """Apply the full contour-refinement config to a resolved agent ``cfg``,
    IN PLACE, choosing the TD3-vs-discrete specific settings from
    ``cfg['agent_type']`` (set by ``resolve_agent_config``). Returns ``cfg``.

    Args:
        cfg: the per-agent cfg from ``resolve_agent_config``.
        baseline_cfg_name: e.g. 'CAMUS/camus.yaml' / 'BRISC/brisc.yaml'
            (the ATTENTION U-Net; Phase B adds label_frac downstream).
        uncertainty_gate: True only where the prob_map diagnostic is USABLE.
        verbose: print a one-line summary (goes to the run log).
    """
    at = str(cfg.get('agent_type', '')).upper()
    if at not in _CONTINUOUS_AGENTS and at not in _DISCRETE_AGENTS:
        raise ValueError(
            f"apply_refinement_config: unrecognised agent_type {at!r} "
            f"(expected one of {_CONTINUOUS_AGENTS | _DISCRETE_AGENTS}). "
            f"Was resolve_agent_config() called first?")

    cfg['baseline_cfg_name'] = baseline_cfg_name
    cfg.update(_SHARED)
    cfg['uncertainty_gate'] = bool(uncertainty_gate)
    cfg.update(_CONTINUOUS if at in _CONTINUOUS_AGENTS else _DISCRETE)

    if verbose:
        kind = 'continuous' if at in _CONTINUOUS_AGENTS else 'discrete'
        print(f"[config] agent={at} ({kind}) base={baseline_cfg_name} "
              f"reward={cfg['reward_mode']} uncertainty_gate={cfg['uncertainty_gate']} "
              f"refinable_gate={cfg['refinable_gate']} disp_px={cfg['disp_px']} "
              f"max_steps={cfg['max_steps']} spatial_head={cfg['spatial_head']} "
              f"bc_warm_start={cfg['bc_warm_start']} bc_demo_max_steps={cfg['bc_demo_max_steps']}")
    return cfg
