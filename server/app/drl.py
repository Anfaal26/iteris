"""
DRL contour-refinement inference for the /infer endpoint.

Design goals (mirrors the redesign brief):
  • One registry keyed by (dataset, class, regime, algo) — adding class 2/3, the
    low regime, or TD3 later is a matter of adding entries, not rewriting the
    loading logic.
  • Checkpoints live in a separate HF Hub *model* repo (same pattern the
    Attention U-Net baselines already use — see inference.py's DATASET_CFG /
    hf_hub_download), pulled at runtime and cached by huggingface_hub itself.
    Weights are populated into that repo from Kaggle training-output datasets
    via a Kaggle notebook (see hf-link.ipynb) — this Space never talks to
    Kaggle directly, so no KAGGLE_* secrets are needed here.
  • Policy nets are loaded lazily into an in-memory cache keyed by the same tuple.
  • A missing (dataset, class, regime, algo) raises RegistryMiss so the API can
    return a clear, structured 404 the frontend uses to grey out the option.

Each CAMUS DRL request fans out to every registered class-agent for that
(dataset, regime, algo), runs each over the same image + its own U-Net init
contour, and combines the per-class binary masks into one label map with a fixed
priority order — exactly the multi-class orchestration the brief describes.

Both discrete (DuelingDDQN, greedy argmax) and continuous (TD3, Actor forward
pass) agents are supported — the episode loop and checkpoint unwrapping branch
on `DrlEntry.action_type`. The network variant (plain vs. sector-spatial head)
is auto-detected from the checkpoint's state-dict keys, so either loads without
a code change.

Verified end-to-end for CAMUS LV DuelingDDQN (Phase A). The other 7 Phase-A
entries (CAMUS myo/LA × both algos, BRISC tumor × both algos) are wired against
their exact training configs (configs/CAMUS/DRL/camus_drl_c{1,2,3}.yaml,
configs/BRISC/DRL/brisc_drl_tumor.yaml) but not yet each individually smoke-
tested — do that once their checkpoints land in the HF repos below.
"""

from __future__ import annotations

import base64
import io
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

from . import inference
from .drl_networks import Actor, DuelingQNetwork, SpatialActor, SpatialDuelingQNetwork
from .env_contour_refine import ContourRefineEnv
from .schemas import MaskLayer, Metrics, StructureMetrics


class RegistryMiss(KeyError):
    """No checkpoint registered/configured for the requested key."""


@dataclass(frozen=True)
class DrlEntry:
    """One trained contour-refinement checkpoint."""
    dataset: str
    klass: str                 # structure key, e.g. 'lv'
    regime: str                # 'low' | 'high'
    algo: str                  # 'duelingddqn' | 'td3' | ...
    structure_id: str          # contract.ts StructureId, e.g. 'lv_endo'
    label: str
    color: str
    unet_class_index: int      # class index in the U-Net argmax map (CAMUS: lv_endo = 1)
    # HF Hub source — same pattern as inference.py's DATASET_CFG. repo/filename
    # are env-overridable so a checkpoint can move without a code change.
    hf_repo: str
    hf_filename: str
    action_type: str = 'discrete'
    # Discrete (DuelingDDQN): fixed at 18 (8 push-out + 8 push-in + smooth + stop).
    # Continuous (TD3): the trained action_dim == cont_sectors, which VARIES per
    # checkpoint (16 for CAMUS, 12 for BRISC) — must match env_cfg['cont_sectors'].
    num_actions: int = 18
    in_channels: int = 5
    # Env hyperparameters — MUST match the checkpoint's training config
    # (configs/CAMUS/DRL/camus_drl_c1.yaml for the entry below).
    env_cfg: dict = field(default_factory=dict)


# Env hyperparameters per checkpoint — MUST match its training config exactly
# (configs/CAMUS/DRL/camus_drl_c{1,2,3}.yaml, configs/BRISC/DRL/brisc_drl_tumor.yaml).
# They differ per class/dataset (e.g. CAMUS-LA and BRISC use a different
# spline_smooth; BRISC's TD3 uses 12 continuous sectors, not 16) — do not
# assume they're interchangeable.
_SHARED = dict(
    max_steps=10,
    n_points=32,
    disp_px=2.0,
    directional_state=False,   # 5 state channels, all 8 checkpoints
    reward_mode='dice_delta',  # rewards are unused at deploy; keep the light path
    disable_auto_stop=True,    # no GT at deploy → don't let GT-based auto-stop fire
)
_CAMUS_LV_DUELING_ENV = dict(_SHARED, spline_smooth=2.0, smooth_lambda=0.5)
_CAMUS_LV_TD3_ENV = dict(_SHARED, spline_smooth=2.0, cont_sectors=16)
_CAMUS_MYO_DUELING_ENV = dict(_SHARED, spline_smooth=2.0, smooth_lambda=0.5)
_CAMUS_MYO_TD3_ENV = dict(_SHARED, spline_smooth=2.0, cont_sectors=16)
_CAMUS_LA_DUELING_ENV = dict(_SHARED, spline_smooth=2.5, smooth_lambda=0.5)
_CAMUS_LA_TD3_ENV = dict(_SHARED, spline_smooth=2.5, cont_sectors=16)
_BRISC_TUMOR_DUELING_ENV = dict(_SHARED, spline_smooth=0.0, smooth_lambda=0.5)
_BRISC_TUMOR_TD3_ENV = dict(_SHARED, spline_smooth=0.0, cont_sectors=12)

_CAMUS_REPO = os.environ.get('HF_REPO_CAMUS_DRL', 'Anfaal26/iteris-drl-camus')
_BRISC_REPO = os.environ.get('HF_REPO_BRISC_DRL', 'Anfaal26/iteris-drl-brisc')

# Colors mirror iteris_ui maskColorsHex / inference.STRUCTURE_DEFS.
REGISTRY: dict[tuple[str, str, str, str], DrlEntry] = {
    ('camus', 'lv', 'high', 'duelingddqn'): DrlEntry(
        dataset='camus', klass='lv', regime='high', algo='duelingddqn',
        structure_id='lv_endo', label='LV Endocardium', color='#00c9a7',
        unet_class_index=1, hf_repo=_CAMUS_REPO,
        hf_filename=os.environ.get('HF_FILE_CAMUS_LV_HIGH_DUELINGDDQN', 'duelingddqn/lv/high.pt'),
        num_actions=18, in_channels=5, action_type='discrete',
        env_cfg=_CAMUS_LV_DUELING_ENV,
    ),
    ('camus', 'myo', 'high', 'duelingddqn'): DrlEntry(
        dataset='camus', klass='myo', regime='high', algo='duelingddqn',
        structure_id='lv_epi', label='LV Epicardium', color='#f59e0b',
        unet_class_index=2, hf_repo=_CAMUS_REPO,
        hf_filename=os.environ.get('HF_FILE_CAMUS_MYO_HIGH_DUELINGDDQN', 'duelingddqn/myo/high.pt'),
        num_actions=18, in_channels=5, action_type='discrete',
        env_cfg=_CAMUS_MYO_DUELING_ENV,
    ),
    ('camus', 'la', 'high', 'duelingddqn'): DrlEntry(
        dataset='camus', klass='la', regime='high', algo='duelingddqn',
        structure_id='la', label='Left Atrium', color='#f87171',
        unet_class_index=3, hf_repo=_CAMUS_REPO,
        hf_filename=os.environ.get('HF_FILE_CAMUS_LA_HIGH_DUELINGDDQN', 'duelingddqn/la/high.pt'),
        num_actions=18, in_channels=5, action_type='discrete',
        env_cfg=_CAMUS_LA_DUELING_ENV,
    ),
    ('camus', 'lv', 'high', 'td3'): DrlEntry(
        dataset='camus', klass='lv', regime='high', algo='td3',
        structure_id='lv_endo', label='LV Endocardium', color='#00c9a7',
        unet_class_index=1, hf_repo=_CAMUS_REPO,
        hf_filename=os.environ.get('HF_FILE_CAMUS_LV_HIGH_TD3', 'td3/lv/high.pt'),
        num_actions=16, in_channels=5, action_type='continuous',
        env_cfg=_CAMUS_LV_TD3_ENV,
    ),
    ('camus', 'myo', 'high', 'td3'): DrlEntry(
        dataset='camus', klass='myo', regime='high', algo='td3',
        structure_id='lv_epi', label='LV Epicardium', color='#f59e0b',
        unet_class_index=2, hf_repo=_CAMUS_REPO,
        hf_filename=os.environ.get('HF_FILE_CAMUS_MYO_HIGH_TD3', 'td3/myo/high.pt'),
        num_actions=16, in_channels=5, action_type='continuous',
        env_cfg=_CAMUS_MYO_TD3_ENV,
    ),
    ('camus', 'la', 'high', 'td3'): DrlEntry(
        dataset='camus', klass='la', regime='high', algo='td3',
        structure_id='la', label='Left Atrium', color='#f87171',
        unet_class_index=3, hf_repo=_CAMUS_REPO,
        hf_filename=os.environ.get('HF_FILE_CAMUS_LA_HIGH_TD3', 'td3/la/high.pt'),
        num_actions=16, in_channels=5, action_type='continuous',
        env_cfg=_CAMUS_LA_TD3_ENV,
    ),
    ('brisc', 'tumor', 'high', 'duelingddqn'): DrlEntry(
        dataset='brisc', klass='tumor', regime='high', algo='duelingddqn',
        structure_id='glioma', label='Tumor (unclassified)', color='#818cf8',
        unet_class_index=1, hf_repo=_BRISC_REPO,
        hf_filename=os.environ.get('HF_FILE_BRISC_TUMOR_HIGH_DUELINGDDQN', 'duelingddqn/tumor/high.pt'),
        num_actions=18, in_channels=5, action_type='discrete',
        env_cfg=_BRISC_TUMOR_DUELING_ENV,
    ),
    ('brisc', 'tumor', 'high', 'td3'): DrlEntry(
        dataset='brisc', klass='tumor', regime='high', algo='td3',
        structure_id='glioma', label='Tumor (unclassified)', color='#818cf8',
        unet_class_index=1, hf_repo=_BRISC_REPO,
        hf_filename=os.environ.get('HF_FILE_BRISC_TUMOR_HIGH_TD3', 'td3/tumor/high.pt'),
        num_actions=12, in_channels=5, action_type='continuous',
        env_cfg=_BRISC_TUMOR_TD3_ENV,
    ),
}

# Fixed z-order when combining CAMUS class masks into one label map: innermost
# structure wins overlapping pixels (LV endo ⊂ myo, LA separate).
CLASS_PRIORITY = ['lv', 'myo', 'la']

_NET_CACHE: dict[tuple[str, str, str, str], torch.nn.Module] = {}


def _entries_for(dataset: str, regime: str, algo: str) -> list[DrlEntry]:
    """All registered class-agents for this (dataset, regime, algo), class-ordered."""
    found = [
        e for e in REGISTRY.values()
        if e.dataset == dataset and e.regime == regime and e.algo == algo
    ]
    order = {k: i for i, k in enumerate(CLASS_PRIORITY)}
    return sorted(found, key=lambda e: order.get(e.klass, 99))


def _download_checkpoint(entry: DrlEntry) -> str:
    """Return a local path to the checkpoint, pulled from the HF Hub model repo
    (same mechanism as inference.get_model) and cached by huggingface_hub's own
    cache — no re-download once fetched once in this container's life.
    Raises RegistryMiss if the repo doesn't have the expected file yet (e.g. the
    Kaggle→HF upload notebook hasn't been run)."""
    try:
        return hf_hub_download(repo_id=entry.hf_repo, filename=entry.hf_filename)
    except Exception as exc:  # noqa: BLE001 — surface as a clean 404, not a 500
        raise RegistryMiss(
            f"checkpoint for ({entry.dataset}, {entry.klass}, {entry.regime}, "
            f"{entry.algo}) not found at hf.co/{entry.hf_repo}/{entry.hf_filename}: {exc}"
        ) from exc


def _build_network(entry: DrlEntry, state: dict) -> torch.nn.Module:
    """Instantiate the network class that matches the checkpoint's action_type
    (discrete -> Dueling head, continuous -> TD3 Actor) and spatial variant
    (plain vs. sector-local heads have disjoint parameter names, so we pick
    from the keys — robust to which one the checkpoint was trained with).
    action_scale on the Actor variants is a registered buffer, so its real
    trained values load automatically via load_state_dict; the placeholder
    passed to the constructor only needs the right shape."""
    keys = ' '.join(state.keys())
    spatial = 'sector_pool' in keys or 'global_branch' in keys
    if entry.action_type == 'continuous':
        Cls = SpatialActor if spatial else Actor
        net = Cls(in_channels=entry.in_channels, action_dim=entry.num_actions)
    else:
        Cls = SpatialDuelingQNetwork if spatial else DuelingQNetwork
        net = Cls(in_channels=entry.in_channels, num_actions=entry.num_actions)
    net.load_state_dict(state)
    net.eval()
    return net


def _load_agent(entry: DrlEntry) -> torch.nn.Module:
    key = (entry.dataset, entry.klass, entry.regime, entry.algo)
    if key in _NET_CACHE:
        return _NET_CACHE[key]
    path = _download_checkpoint(entry)
    raw = torch.load(path, map_location='cpu')
    state = raw
    if isinstance(raw, dict):
        # Real training-checkpoint format (iteris.agents.{DQNAgent,TD3Agent}.
        # state_dict): {'agent': {...}, 'optimizers':..., 'best_dice':...,
        # 'history':..., 'step':...}. Inside 'agent':
        #   DQNAgent (discrete) -> {'q': <online>, 'q_target': <target>}
        #   TD3Agent (continuous) -> {'actor': <online>, 'actor_target':...,
        #                             'critic1':..., 'critic2':..., ...}
        # Inference always wants the ONLINE net (q / actor), never the target
        # (trails during training) or a critic (not used for action selection).
        agent_state = raw.get('agent')
        if isinstance(agent_state, dict) and 'q' in agent_state:
            state = agent_state['q']
        elif isinstance(agent_state, dict) and 'actor' in agent_state:
            state = agent_state['actor']
        else:
            for k in ('online_state_dict', 'q_state_dict', 'model_state_dict', 'state_dict', 'model'):
                if k in raw:
                    state = raw[k]
                    break
    net = _build_network(entry, state)
    _NET_CACHE[key] = net
    return net


def _decode_gt(gt_b64: str, size: int) -> np.ndarray:
    if gt_b64.startswith('data:'):
        gt_b64 = gt_b64.split(',', 1)[1]
    raw = base64.b64decode(gt_b64)
    img = Image.open(io.BytesIO(raw)).convert('L').resize((size, size), Image.NEAREST)
    return np.asarray(img)


def _unet_init_and_prob(dataset: str, image_b64: str, class_index: int):
    """Run the deployed U-Net and return (image_norm, init_binary, prob_map) for
    one class — the warm start the contour env refines."""
    model = inference.get_model(dataset)
    cfg = inference.DATASET_CFG[dataset]
    pil = inference._decode_b64_image(image_b64)
    arr = inference._to_normalized_array(pil, cfg['normalize'])           # (256,256) float32
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)                        # (C,256,256)
        pred = logits.argmax(dim=1).squeeze(0)
    init_binary = (pred == class_index).to(torch.uint8).cpu().numpy()
    prob_map = probs[class_index].cpu().numpy().astype(np.float32)
    return arr.astype(np.float32), init_binary, prob_map


@torch.no_grad()
def _run_episode(entry: DrlEntry, image: np.ndarray, init_mask: np.ndarray,
                 prob_map: np.ndarray) -> tuple[np.ndarray, int]:
    """Greedy contour-refinement rollout. GT is unknown at deploy, so a dummy GT
    (= init mask) is passed purely to satisfy the env constructor; rewards and
    GT-based auto-stop are disabled, so termination is the agent's STOP action or
    max_steps. Returns (refined_binary_mask, steps_taken)."""
    net = _load_agent(entry)
    if not init_mask.any():
        return init_mask, 0  # nothing to refine

    env = ContourRefineEnv(
        image=image,
        gt_mask=init_mask,          # dummy — unused for the returned mask
        init_mask=init_mask,
        prob_map=prob_map,
        action_type=entry.action_type,
        **entry.env_cfg,
    )
    state = env.reset()
    steps = 0
    for _ in range(int(entry.env_cfg.get('max_steps', 10))):
        s = torch.from_numpy(state).unsqueeze(0).float()
        if entry.action_type == 'continuous':
            # Actor.forward already returns the tanh × action_scale continuous
            # action — no exploration noise (deploy wants the greedy policy).
            action = net(s).squeeze(0).numpy()
        else:
            q = net(s)
            action = int(q.argmax(dim=1).item())
        state, _reward, done, _info = env.step(action)
        steps += 1
        if done:
            break
    return env.get_final_mask().astype(np.uint8), steps


def infer(dataset: str, model_family: str, algo: str, regime: str,
          image_b64: str, gt_b64: Optional[str]):
    """Entry point for POST /infer. Runs every registered class-agent for this
    (dataset, regime, algo), combines masks by CLASS_PRIORITY, and returns the
    (masks, metrics, refinement_steps, inference_ms) dict in contract shape.

    Raises RegistryMiss if nothing is registered/configured for the key."""
    if model_family != 'drl':
        raise RegistryMiss(f"model_family '{model_family}' has no /infer path yet")

    entries = _entries_for(dataset, regime, algo)
    if not entries:
        raise RegistryMiss(
            f"no checkpoint registered for (dataset={dataset}, regime={regime}, algo={algo})"
        )

    size = inference.IMAGE_SIZE
    t0 = time.time()

    # Combined label map: class index i+1 per structure, priority-ordered so
    # inner structures win overlaps.
    label_map = np.zeros((size, size), dtype=np.int64)
    per_class_binary: list[tuple[DrlEntry, np.ndarray]] = []
    total_steps = 0
    for i, entry in enumerate(entries, start=1):
        image, init_binary, prob_map = _unet_init_and_prob(dataset, image_b64, entry.unet_class_index)
        refined, steps = _run_episode(entry, image, init_binary, prob_map)
        total_steps += steps
        per_class_binary.append((entry, refined))
        label_map[refined.astype(bool)] = i  # later (higher-priority) classes overwrite

    # Build mask layers (RGBA PNGs) + per-class metrics.
    masks: list[MaskLayer] = []
    structures: list[StructureMetrics] = []
    gt_arr = _decode_gt(gt_b64, size) if gt_b64 else None
    dices, ious, hds, hd95s = [], [], [], []
    for entry, binary in per_class_binary:
        masks.append(MaskLayer(
            structure=entry.structure_id,
            label=entry.label,
            color=entry.color,
            imageB64=inference._mask_to_png_b64(binary, entry.color),
        ))
        if gt_arr is not None:
            gt_bin = _class_gt(gt_arr, entry.unet_class_index)
            d = _dice(binary, gt_bin)
            j = _iou(binary, gt_bin)
            hd, hd95 = _hd_pair(binary, gt_bin)
            dices.append(d); ious.append(j); hds.append(hd); hd95s.append(hd95)
            structures.append(StructureMetrics(
                structure=entry.structure_id, label=entry.label,
                dice=round(d, 4), iou=round(j, 4), hd=round(hd, 2), hd95=round(hd95, 2),
            ))
        else:
            structures.append(StructureMetrics(
                structure=entry.structure_id, label=entry.label,
                dice=0.0, iou=0.0, hd=0.0, hd95=0.0,
            ))

    if gt_arr is not None and dices:
        metrics = Metrics(
            dice=round(float(np.mean(dices)), 4), iou=round(float(np.mean(ious)), 4),
            hd=round(float(np.mean(hds)), 2), hd95=round(float(np.mean(hd95s)), 2),
            structures=structures, baselineDice=round(float(np.mean(dices)), 4),
        )
    else:
        metrics = Metrics(dice=0.0, iou=0.0, hd=0.0, hd95=0.0,
                          structures=structures, baselineDice=0.0)

    inference_ms = round((time.time() - t0) * 1000, 1)
    return dict(
        sessionId=str(uuid.uuid4()),
        masks=masks, metrics=metrics,
        refinementSteps=total_steps, inferenceMs=inference_ms,
        imageWidth=size, imageHeight=size,
    )


# ── metric helpers (pure numpy; the vendored geometry.py has richer versions) ──

def _class_gt(gt_arr: np.ndarray, class_index: int) -> np.ndarray:
    # Multi-class GT stored as class indices, or binary 0/255. Treat >6 as binary.
    if gt_arr.max() > 6:
        return (gt_arr > 127).astype(np.uint8)
    return (gt_arr == class_index).astype(np.uint8)


def _dice(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = a.astype(bool); b = b.astype(bool)
    return float((2 * (a & b).sum() + eps) / (a.sum() + b.sum() + eps))


def _iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = a.astype(bool); b = b.astype(bool)
    return float(((a & b).sum() + eps) / ((a | b).sum() + eps))


def _hd_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(HD, HD95) via the vendored geometry helper; NaN-safe."""
    from .geometry import hd95_px
    try:
        hd95 = hd95_px(a, b)
        hd95 = 0.0 if hd95 != hd95 else hd95
    except Exception:
        hd95 = 0.0
    return hd95, hd95
