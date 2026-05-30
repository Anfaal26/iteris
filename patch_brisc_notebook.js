'use strict';
const fs = require('fs');

function src(s) {
  s = s.replace(/^\n/, '');
  const lines = s.split('\n');
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    if (i < lines.length - 1) out.push(lines[i] + '\n');
    else if (lines[i] !== '') out.push(lines[i]);
  }
  return out;
}

function patchCell(nb, id, fn) {
  const cell = nb.cells.find(c => c.id === id);
  if (!cell) { console.error('MISSING CELL:', id); process.exit(1); }
  if (typeof fn === 'string') cell.source = src(fn);
  else cell.source = fn(cell.source);
}

function patchNotebook(fpath, patches) {
  const nb = JSON.parse(fs.readFileSync(fpath, 'utf8'));
  for (const [id, fn] of Object.entries(patches)) patchCell(nb, id, fn);
  fs.writeFileSync(fpath, JSON.stringify(nb, null, 1) + '\n', 'utf8');
  console.log('✓', fpath);
}

function updateAgentComment(source) {
  return source.map(line =>
    /^AGENT_NAME\s*=/.test(line)
      ? "AGENT_NAME = 'DDQN'   # DQN | DDQN | DUELING | DDPG | DQN_TRACE | DUELING_TRACE\n"
      : line
  );
}

function updateMdTitle(source) {
  return source.map(line =>
    line.includes('MSA-DUELING')
      ? line.replace(
          /`'MSA-DUELING'`\s*·\s*`'DDPG'`/,
          "`'DDPG'` · `'DQN_TRACE'` · `'DUELING_TRACE'`"
        )
      : line
  );
}

// ── BRISC cell templates ──────────────────────────────────────────────────────

const VIZSETUP = `
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from iteris.env import SegmentationEnv, dice_score, hd95_px
from iteris.drl_training import ENV_REGISTRY

CLASS_IDX   = cfg['target_class']      # 1
CLASS_NAME  = cfg['class_name']        # 'tumor'
CLASS_COLOR = '#F43F5E'
CMAP_TUMOR  = ListedColormap([CLASS_COLOR])

IS_TRACING = cfg.get('env_class') == 'contour_tracing'

if IS_TRACING:
    import iteris.contour_viz as cv
    ENV_KW  = cv.trace_env_kwargs(cfg)
    replays = cv.build_trace_replays(agent, test_samples, ENV_KW, n_viz=8, seed=0)
    print(f'Tracing — {len(replays)} replays. Dice gain: {replays[0]["gain"]:+.4f} → {replays[-1]["gain"]:+.4f}')
else:
    def dice_to_iou(d): return d / (2.0 - d + 1e-8)

    ENV_CLS        = ENV_REGISTRY.get(cfg.get('env_class', 'default'), SegmentationEnv)
    DISCRETE_NAMES = ENV_CLS.DISCRETE_NAMES
    print(f'Env: {ENV_CLS.__name__}  ({len(DISCRETE_NAMES)} actions)')

    ENV_KW = dict(
        action_type      = agent.action_type,
        max_steps        = cfg.get('max_steps', 20),
        shift_px         = cfg.get('shift_px', 1),
        sdt_clip         = cfg.get('sdt_clip', 15.0),
        reward_clip      = cfg.get('reward_clip', 1.0),
        cont_morph_scale = cfg.get('cont_morph_scale', 0.15),
        cont_trans_scale = cfg.get('cont_trans_scale', 0.01),
        reward_mode      = cfg.get('reward_mode', 'iou_delta'),
        reward_alpha     = cfg.get('reward_alpha', 1.0),
        reward_beta      = cfg.get('reward_beta',  0.0),
        hd_norm          = cfg.get('hd_norm', 17.0),
        stop_eps_dice    = cfg.get('stop_eps_dice', 0.001),
        stop_eps_hd      = cfg.get('stop_eps_hd', 0.5),
        stop_n           = cfg.get('stop_n', 3),
    )

    def replay_episode(sample):
        """Greedy rollout. Returns (masks, dice_list, iou_list, action_list)."""
        env = SegmentationEnv(sample['image'], sample['gt_mask'], sample['init_mask'], **ENV_KW)
        state = env.reset()
        masks, actions = [env.mask.copy()], []
        while True:
            if agent.action_type == 'discrete':
                a = agent.select_action(state, epsilon=0.0)
            else:
                a = agent.select_action(state, explore=False)
            actions.append(a)
            state, _, done, _ = env.step(a)
            masks.append(env.mask.copy())
            if done:
                break
        dices = list(env.dice_history)
        ious  = [dice_to_iou(d) for d in dices]
        return masks, dices, ious, actions

    N_VIZ = min(8, len(test_samples))
    np.random.seed(0)
    viz_idx = np.random.choice(len(test_samples), size=N_VIZ, replace=False).tolist()
    print(f'Replaying {N_VIZ} test samples...')
    replays = []
    for i in viz_idx:
        masks, dices, ious, actions = replay_episode(test_samples[i])
        replays.append(dict(
            sample    = test_samples[i],
            masks     = masks, dices = dices, ious = ious, actions = actions,
            init_d    = dices[0],  final_d   = dices[-1],
            init_iou  = ious[0],   final_iou = ious[-1],
            gain_dice = dices[-1] - dices[0],
            gain_iou  = ious[-1]  - ious[0],
        ))
    replays.sort(key=lambda r: r['gain_iou'])
    print(f'Done. Delta-IoU: {replays[0]["gain_iou"]:+.4f} to {replays[-1]["gain_iou"]:+.4f}')
    print(f'      Delta-Dice: {replays[0]["gain_dice"]:+.4f} to {replays[-1]["gain_dice"]:+.4f}')
`;

const COMPARE = `
if IS_TRACING:
    fig = cv.plot_trace_comparison(
        replays, baseline_cfg, cfg, CLASS_IDX, CLASS_NAME,
        out_path=f'/kaggle/working/brisc_{cfg["agent_type"].lower()}_comparison.png')
    plt.show()
else:
    picks = [('BEST gain', replays[-1]), ('MEDIAN gain', replays[N_VIZ//2]), ('WORST gain', replays[0])]
    fig, axes = plt.subplots(len(picks), 4, figsize=(18, 4.5*len(picks)))
    for row, (label, r) in enumerate(picks):
        s          = r['sample']
        final_mask = r['masks'][-1]
        img, init, gt = s['image'], s['init_mask'], s['gt_mask']
        hd_init  = hd95_px(init, gt)
        hd_final = hd95_px(final_mask, gt)
        cells = [
            ('Input (MRI)',                   None,       None,        None),
            ('U-Net init',                    init,       r['init_d'], hd_init),
            (f'{cfg["agent_type"]} refined',  final_mask, r['final_d'], hd_final),
            ('Ground Truth',                  gt,         1.0,         0.0),
        ]
        for col, (title, mask, d, h) in enumerate(cells):
            ax = axes[row, col]
            ax.imshow(img, cmap='gray')
            if mask is not None:
                ax.imshow(np.ma.masked_where(mask == 0, mask), cmap=CMAP_TUMOR, alpha=0.5)
            if d is not None and h is not None:
                iou = dice_to_iou(d)
                ax.set_title(f'{title}\\nDice {d:.3f}  IoU {iou:.3f}  HD95 {h:.1f}px', fontsize=9)
            else:
                ax.set_title(f'{title}\\n[{label}]  dDice {r["gain_dice"]:+.4f}  dIoU {r["gain_iou"]:+.4f}',
                             fontsize=9)
            ax.axis('off')
    plt.suptitle(f'BRISC {cfg["agent_type"]} — Tumour Refinement', fontsize=13)
    plt.tight_layout()
    out = f'/kaggle/working/brisc_{cfg["agent_type"].lower()}_comparison.png'
    plt.savefig(out, dpi=150); plt.show(); print(f'Saved to {out}')
`;

const PLAYBACK = `
if IS_TRACING:
    fig = cv.plot_trajectory_playback(
        replays[-1], cfg, CLASS_NAME,
        out_path=f'/kaggle/working/brisc_{cfg["agent_type"].lower()}_playback.png')
    plt.show()
else:
    from scipy import ndimage
    r = replays[-1]; s = r['sample']
    gt_edge = s['gt_mask'] ^ ndimage.binary_erosion(s['gt_mask'], iterations=1)
    n_steps = len(r['masks'])
    ncols = 5; nrows = int(np.ceil(n_steps / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 3.2*nrows))
    axes = np.atleast_2d(axes).flatten()
    for i, (mask, dice, iou) in enumerate(zip(r['masks'], r['dices'], r['ious'])):
        ax = axes[i]
        ax.imshow(s['image'], cmap='gray')
        ax.imshow(np.ma.masked_where(mask == 0, mask), cmap=CMAP_TUMOR, alpha=0.55)
        ax.imshow(np.ma.masked_where(~gt_edge.astype(bool), gt_edge), cmap='Greens', alpha=0.8)
        if i == 0:
            title = f't=0 (init)\\nDice {dice:.3f}  IoU {iou:.3f}'
        else:
            a = r['actions'][i - 1]
            if agent.action_type == 'discrete':
                a_str = DISCRETE_NAMES[int(a)]
            else:
                a_str = f'm={a[0]:+.3f} dy={a[1]*256:+.1f}px dx={a[2]*256:+.1f}px'
            title = f't={i}  {a_str}\\nDice {dice:.3f}  D{dice-r["dices"][i-1]:+.4f}'
        ax.set_title(title, fontsize=7); ax.axis('off')
    for j in range(n_steps, len(axes)): axes[j].axis('off')
    plt.suptitle(f'{cfg["agent_type"]} playback — tumor  (green = GT boundary)', fontsize=12)
    plt.tight_layout()
    out = f'/kaggle/working/brisc_{cfg["agent_type"].lower()}_playback.png'
    plt.savefig(out, dpi=150); plt.show(); print(f'Saved to {out}')
`;

const BEHAVIOUR = `
if IS_TRACING:
    fig = cv.plot_direction_behaviour(
        replays, cfg, CLASS_NAME,
        out_path=f'/kaggle/working/brisc_{cfg["agent_type"].lower()}_behaviour.png')
    plt.show()
else:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    for r in replays: ax.plot(r['dices'], alpha=0.5, lw=1)
    max_len = max(len(r['dices']) for r in replays)
    padded  = np.stack([np.pad(r['dices'], (0, max_len - len(r['dices'])),
                               constant_values=r['dices'][-1]) for r in replays])
    ax.plot(padded.mean(axis=0), color='black', lw=2.5, label='mean')
    ax.set(xlabel='Step', ylabel='Dice', title='Tumor Dice trajectory')
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    iou_padded = np.stack([np.pad(r['ious'], (0, max_len - len(r['ious'])),
                                  constant_values=r['ious'][-1]) for r in replays])
    for r in replays: ax.plot(r['ious'], alpha=0.5, lw=1)
    ax.plot(iou_padded.mean(axis=0), color='black', lw=2.5, label='mean')
    ax.set(xlabel='Step', ylabel='IoU', title='Tumor IoU trajectory (reward signal)')
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[2]
    all_actions = [a for r in replays for a in r['actions']]
    if agent.action_type == 'discrete':
        counts = np.bincount(all_actions, minlength=len(DISCRETE_NAMES))
        bars = ax.bar(range(len(DISCRETE_NAMES)), counts / counts.sum())
        bars[-1].set_color('#22C55E')
        ax.set_xticks(range(len(DISCRETE_NAMES)))
        ax.set_xticklabels(DISCRETE_NAMES, rotation=30)
        ax.set(ylabel='Frequency', title='Action distribution'); ax.grid(alpha=0.3, axis='y')
    else:
        arr = np.array(all_actions)
        ax.hist(arr[:, 0], bins=30, alpha=0.7, color=CLASS_COLOR, label='morph')
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.set(xlabel='Morph action', title='DDPG morph distribution')
        ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle(f'BRISC {cfg["agent_type"]} — tumor behaviour')
    plt.tight_layout()
    out = f'/kaggle/working/brisc_{cfg["agent_type"].lower()}_behaviour.png'
    plt.savefig(out, dpi=150); plt.show()
    print(f'\\n-- {cfg["agent_type"]} tumor behaviour --')
    print(f'  Init  Dice avg : {np.mean([r["init_d"]    for r in replays]):.4f}')
    print(f'  Final Dice avg : {np.mean([r["final_d"]   for r in replays]):.4f}')
    print(f'  Delta Dice avg : {np.mean([r["gain_dice"] for r in replays]):+.4f}')
    print(f'  Final IoU  avg : {np.mean([r["final_iou"] for r in replays]):.4f}')
    print(f'  Delta IoU  avg : {np.mean([r["gain_iou"]  for r in replays]):+.4f}')
    print(f'  Avg ep length  : {np.mean([len(r["dices"]) - 1 for r in replays]):.1f} steps')
`;

const CURVES = `
if IS_TRACING:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(history['step'], history['final_dice_mean'], label='Trace Dice', color=CLASS_COLOR)
    ax1.set(xlabel='Training step', ylabel='Mean Val Dice', title='Boundary-tracing learning curve — tumor')
    ax1.legend(); ax1.grid(alpha=0.3)
    if 'closure_rate' in history.columns:
        ax2.plot(history['step'], history['closure_rate'], color='C2')
        ax2.set(xlabel='Training step', ylabel='Closure rate',
                title=f'Closure rate — {cfg["agent_type"]} (tumor)')
    ax2.grid(alpha=0.3)
    plt.suptitle(f'BRISC {cfg["agent_type"]}_TRACE — tumor learning curves')
    plt.tight_layout()
    out = f'/kaggle/working/brisc_{cfg["agent_type"].lower()}_curves.png'
    plt.savefig(out, dpi=150); plt.show(); print(f'Saved to {out}')
else:
    def dice_to_iou(d): return d / (2.0 - d + 1e-8)
    init_iou_hist  = history['init_dice_mean'].apply(dice_to_iou)
    final_iou_hist = history['final_dice_mean'].apply(dice_to_iou)
    delta_iou_hist = final_iou_hist - init_iou_hist
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['step'], init_iou_hist,  label='Init IoU (U-Net)', ls='--', alpha=0.6)
    ax1.plot(history['step'], final_iou_hist, label=f'Final IoU ({cfg["agent_type"]})', color=CLASS_COLOR)
    ax1.set(xlabel='Training step', ylabel='Mean Val IoU', title='BRISC — Refinement curve (IoU)')
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(history['step'], delta_iou_hist, color=CLASS_COLOR)
    ax2.axhline(0, color='k', lw=0.8)
    ax2.fill_between(history['step'], delta_iou_hist, 0,
                     where=(delta_iou_hist > 0), alpha=0.15, color='green')
    ax2.fill_between(history['step'], delta_iou_hist, 0,
                     where=(delta_iou_hist < 0), alpha=0.15, color='red')
    ax2.set(xlabel='Training step', ylabel='Delta IoU',
            title=f'Refinement gain — {cfg["agent_type"]} (tumor)')
    ax2.grid(alpha=0.3)
    plt.suptitle(f'BRISC {cfg["agent_type"]} — tumor learning curves')
    plt.tight_layout()
    out = f'/kaggle/working/brisc_{cfg["agent_type"].lower()}_curves.png'
    plt.savefig(out, dpi=150); plt.show(); print(f'Saved to {out}')
`;

const TEST = `
import json

if IS_TRACING:
    test_metrics = cv.evaluate_trace_testset(agent, test_samples, ENV_KW)
    print(json.dumps(test_metrics, indent=2))
    summary = {**test_metrics,
               'agent_type':    cfg['agent_type'],
               'target_class':  cfg['target_class'],
               'class_name':    cfg['class_name'],
               'paradigm':      'contour_tracing',
               'baseline_dice': 0.8351,
               'baseline_hd95': 8.36}
    out = f'/kaggle/working/brisc_{cfg["agent_type"].lower()}_test_results.json'
    with open(out, 'w') as f: json.dump(summary, f, indent=2)
    print(f'\\nSaved to {out}')
    print(json.dumps(summary, indent=2))
else:
    from iteris.drl_training import evaluate_agent
    test_metrics = evaluate_agent(agent, test_samples, env_kwargs=ENV_KW)
    test_hd95s = []
    print('Computing HD95 on test samples...')
    for s in test_samples:
        env  = SegmentationEnv(s['image'], s['gt_mask'], s['init_mask'], **ENV_KW)
        state = env.reset()
        while True:
            if agent.action_type == 'discrete':
                a = agent.select_action(state, epsilon=0.0)
            else:
                a = agent.select_action(state, explore=False)
            state, _, done, _ = env.step(a)
            if done:
                break
        test_hd95s.append(hd95_px(env.mask, s['gt_mask']))
    valid_hd95 = [h for h in test_hd95s if not np.isnan(h)]
    test_metrics['final_hd95_mean'] = float(np.mean(valid_hd95)) if valid_hd95 else float('nan')
    test_metrics['init_iou_mean']   = float(dice_to_iou(test_metrics['init_dice_mean']))
    test_metrics['final_iou_mean']  = float(dice_to_iou(test_metrics['final_dice_mean']))
    test_metrics['delta_iou_mean']  = float(test_metrics['final_iou_mean'] - test_metrics['init_iou_mean'])
    summary = {**test_metrics,
               'agent_type':    cfg['agent_type'],
               'target_class':  cfg['target_class'],
               'class_name':    cfg['class_name'],
               'reward_mode':   cfg['reward_mode'],
               'best_val_iou':  float(best_dice),
               'baseline_dice': 0.8351,
               'baseline_hd95': 8.36}
    out = f'/kaggle/working/brisc_{cfg["agent_type"].lower()}_test_results.json'
    with open(out, 'w') as f: json.dump(summary, f, indent=2)
    print(f'\\nSaved to {out}')
    print(json.dumps(summary, indent=2))
`;

// ── Patch BRISC notebook ──────────────────────────────────────────────────────

patchNotebook('D:\\iteris\\notebooks\\04_brisc_drl.ipynb', {
  'md-title':       updateMdTitle,
  'code-config':    updateAgentComment,
  'code-vizsetup':  VIZSETUP,
  'code-compare':   COMPARE,
  'code-playback':  PLAYBACK,
  'code-behaviour': BEHAVIOUR,
  'code-curves':    CURVES,
  'code-test':      TEST,
});

console.log('Done — BRISC notebook patched.');
