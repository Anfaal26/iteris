'use strict';
const fs = require('fs');

// Convert Python source string → Jupyter source array
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

// ── Line-level helpers ────────────────────────────────────────────────────────

// Update the AGENT_NAME comment line only
function updateAgentComment(source) {
  return source.map(line =>
    /^AGENT_NAME\s*=/.test(line)
      ? "AGENT_NAME = 'DDQN'   # DQN | DDQN | DUELING | DDPG | DQN_TRACE | DUELING_TRACE\n"
      : line
  );
}

// Update md-title options table (remove MSA-DUELING, add TRACE variants)
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

// ── CAMUS cell templates (same for 03a / 03b / 03c) ─────────────────────────

const VIZSETUP = `
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from iteris.env import SegmentationEnv, dice_score, hd95_px

CLASS_IDX   = cfg['target_class']
CLASS_NAME  = cfg['class_name']
CLASS_COLOR = baseline_cfg['class_colors'][CLASS_IDX]
CMAP_SINGLE = ListedColormap([CLASS_COLOR])

IS_TRACING = cfg.get('env_class') == 'contour_tracing'

if IS_TRACING:
    import iteris.contour_viz as cv
    ENV_KW  = cv.trace_env_kwargs(cfg)
    replays = cv.build_trace_replays(agent, test_samples, ENV_KW, n_viz=8, seed=0)
    print(f'Tracing — {len(replays)} replays. Dice gain: {replays[0]["gain"]:+.4f} → {replays[-1]["gain"]:+.4f}')
else:
    DISCRETE_NAMES = ['dil-N','dil-E','dil-S','dil-W','ero-N','ero-E','ero-S','ero-W',
                      'sh-\\u2191','sh-\\u2193','sh-\\u2190','sh-\\u2192','no-op']

    ENV_KW = dict(
        action_type      = agent.action_type,
        max_steps        = cfg.get('max_steps', 20),
        shift_px         = cfg.get('shift_px', 2),
        sdt_clip         = cfg.get('sdt_clip', 20.0),
        reward_clip      = cfg.get('reward_clip', 1.0),
        cont_morph_scale = cfg.get('cont_morph_scale', 0.25),
        cont_trans_scale = cfg.get('cont_trans_scale', 0.02),
        reward_mode      = cfg.get('reward_mode', 'dice_delta'),
        reward_alpha     = cfg.get('reward_alpha', 0.5),
        reward_beta      = cfg.get('reward_beta',  0.5),
        hd_norm          = cfg.get('hd_norm', 50.0),
        stop_eps_dice    = cfg.get('stop_eps_dice', 0.001),
        stop_eps_hd      = cfg.get('stop_eps_hd', 0.5),
        stop_n           = cfg.get('stop_n', 3),
    )

    def replay_episode(sample):
        """Greedy rollout. Returns (masks, dice_list, hd95_list, action_list)."""
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
        return masks, list(env.dice_history), list(env.hd95_history), actions

    N_VIZ = min(8, len(test_samples))
    np.random.seed(0)
    viz_idx = np.random.choice(len(test_samples), size=N_VIZ, replace=False).tolist()
    print(f'Replaying {N_VIZ} test samples...')
    replays = []
    for i in viz_idx:
        masks, dices, hd95s, actions = replay_episode(test_samples[i])
        replays.append(dict(sample=test_samples[i], masks=masks, dices=dices,
                            hd95s=hd95s, actions=actions,
                            init_d=dices[0], final_d=dices[-1], gain=dices[-1]-dices[0]))
    replays.sort(key=lambda r: r['gain'])
    print(f'Done. Delta-Dice range: {replays[0]["gain"]:+.4f} to {replays[-1]["gain"]:+.4f}')
`;

const COMPARE = `
if IS_TRACING:
    fig = cv.plot_trace_comparison(
        replays, baseline_cfg, cfg, CLASS_IDX, CLASS_NAME,
        out_path=f'/kaggle/working/{cfg["agent_type"].lower()}_c{CLASS_IDX}_comparison.png')
    plt.show()
else:
    picks = [('BEST gain', replays[-1]), ('MEDIAN gain', replays[N_VIZ//2]), ('WORST gain', replays[0])]
    fig, axes = plt.subplots(len(picks), 4, figsize=(16, 4*len(picks)))
    for row, (label, r) in enumerate(picks):
        s, final_mask = r['sample'], r['masks'][-1]
        img, init, gt = s['image'], s['init_mask'], s['gt_mask']
        cells = [
            ('Input',                        None,       None,        None),
            ('U-Net init',                   init,       r['init_d'], hd95_px(init, gt)),
            (f'{cfg["agent_type"]} refined', final_mask, r['final_d'], hd95_px(final_mask, gt)),
            ('Ground Truth',                 gt,         1.0,         0.0),
        ]
        for col, (title, mask, d, h) in enumerate(cells):
            ax = axes[row, col]
            ax.imshow(img, cmap='gray')
            if mask is not None:
                ax.imshow(np.ma.masked_where(mask==0, mask), cmap=CMAP_SINGLE, alpha=0.5)
            ax.set_title(f'{title}\\nDice {d:.3f}  HD95 {h:.1f}px' if d is not None
                         else f'{title}\\n[{label}]  Delta-Dice {r["gain"]:+.4f}', fontsize=10)
            ax.axis('off')
    plt.suptitle(f'CAMUS {cfg["agent_type"]} — {CLASS_NAME}', fontsize=13)
    plt.tight_layout()
    out = f'/kaggle/working/{cfg["agent_type"].lower()}_c{CLASS_IDX}_comparison.png'
    plt.savefig(out, dpi=150); plt.show(); print(f'Saved to {out}')
`;

const PLAYBACK = `
if IS_TRACING:
    fig = cv.plot_trajectory_playback(
        replays[-1], cfg, CLASS_NAME,
        out_path=f'/kaggle/working/{cfg["agent_type"].lower()}_c{CLASS_IDX}_playback.png')
    plt.show()
else:
    from scipy import ndimage
    r = replays[-1]; s = r['sample']
    gt_edge = s['gt_mask'] ^ ndimage.binary_erosion(s['gt_mask'], iterations=1)
    n_steps = len(r['masks'])
    ncols = 5; nrows = int(np.ceil(n_steps / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = np.atleast_2d(axes).flatten()
    for i, (mask, dice, hd) in enumerate(zip(r['masks'], r['dices'], r['hd95s'])):
        ax = axes[i]
        ax.imshow(s['image'], cmap='gray')
        ax.imshow(np.ma.masked_where(mask==0, mask), cmap=CMAP_SINGLE, alpha=0.55)
        ax.imshow(np.ma.masked_where(~gt_edge.astype(bool), gt_edge), cmap='Greens', alpha=0.8)
        if i == 0:
            title = f't=0 (init)\\nDice {dice:.3f}'
        else:
            a = r['actions'][i-1]
            if agent.action_type == 'discrete':
                a_str = DISCRETE_NAMES[int(a)]
            else:
                a_str = (f'm={a[0]:+.3f} dy={a[1]*256:+.1f}px dx={a[2]*256:+.1f}px')
            title = f't={i}  {a_str}\\nDice {dice:.3f}  D{dice-r["dices"][i-1]:+.4f}'
        ax.set_title(title, fontsize=7); ax.axis('off')
    for j in range(n_steps, len(axes)): axes[j].axis('off')
    plt.suptitle(f'{cfg["agent_type"]} playback — {CLASS_NAME}  (green = GT boundary)', fontsize=12)
    plt.tight_layout()
    out = f'/kaggle/working/{cfg["agent_type"].lower()}_c{CLASS_IDX}_playback.png'
    plt.savefig(out, dpi=150); plt.show(); print(f'Saved to {out}')
`;

const BEHAVIOUR = `
if IS_TRACING:
    fig = cv.plot_direction_behaviour(
        replays, cfg, CLASS_NAME,
        out_path=f'/kaggle/working/{cfg["agent_type"].lower()}_c{CLASS_IDX}_behaviour.png')
    plt.show()
else:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    for r in replays: ax.plot(r['dices'], alpha=0.5, lw=1)
    max_len = max(len(r['dices']) for r in replays)
    padded = np.stack([np.pad(r['dices'], (0, max_len-len(r['dices'])),
                              constant_values=r['dices'][-1]) for r in replays])
    ax.plot(padded.mean(axis=0), color='black', lw=2.5, label='mean')
    ax.set(xlabel='Step', ylabel='Dice', title=f'{CLASS_NAME} Dice trajectory')
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    for r in replays:
        valid = [v for v in r['hd95s'] if not np.isnan(v)]
        ax.plot(valid, alpha=0.5, lw=1)
    ax.set(xlabel='Step', ylabel='HD95 (px)', title=f'{CLASS_NAME} HD95 trajectory')
    ax.grid(alpha=0.3)
    ax = axes[2]
    all_actions = [a for r in replays for a in r['actions']]
    if agent.action_type == 'discrete':
        counts = np.bincount(all_actions, minlength=len(DISCRETE_NAMES))
        ax.bar(range(len(DISCRETE_NAMES)), counts / counts.sum())
        ax.set_xticks(range(len(DISCRETE_NAMES)))
        ax.set_xticklabels(DISCRETE_NAMES, rotation=30)
        ax.set(ylabel='Frequency', title='Action distribution'); ax.grid(alpha=0.3, axis='y')
    else:
        arr = np.array(all_actions)
        ax.hist(arr[:, 0], bins=30, alpha=0.7, label='morph')
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.set(xlabel='Morph action', title='DDPG morph distribution')
        ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle(f'{cfg["agent_type"]} behaviour — {CLASS_NAME}')
    plt.tight_layout()
    out = f'/kaggle/working/{cfg["agent_type"].lower()}_c{CLASS_IDX}_behaviour.png'
    plt.savefig(out, dpi=150); plt.show()
    print(f'\\n-- {cfg["agent_type"]} {CLASS_NAME} behaviour --')
    print(f'  Init  Dice avg : {np.mean([r["init_d"]  for r in replays]):.4f}')
    print(f'  Final Dice avg : {np.mean([r["final_d"] for r in replays]):.4f}')
    print(f'  Delta Dice avg : {np.mean([r["gain"]    for r in replays]):+.4f}')
    print(f'  Avg ep length  : {np.mean([len(r["dices"])-1 for r in replays]):.1f} steps')
`;

const CURVES = `
if IS_TRACING:
    # Tracing history: step, final_dice_mean, final_hd95_mean, closure_rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(history['step'], history['final_dice_mean'], label='Trace Dice', color='C1')
    ax1.set(xlabel='Training step', ylabel='Mean Val Dice', title='Boundary-tracing learning curve')
    ax1.legend(); ax1.grid(alpha=0.3)
    if 'closure_rate' in history.columns:
        ax2.plot(history['step'], history['closure_rate'], color='C2')
        ax2.set(xlabel='Training step', ylabel='Closure rate',
                title=f'Closure rate — {cfg["agent_type"]} ({CLASS_NAME})')
    ax2.grid(alpha=0.3)
    plt.suptitle(f'CAMUS {cfg["agent_type"]}_TRACE — {CLASS_NAME} learning curves')
    plt.tight_layout()
    out = f'/kaggle/working/{cfg["agent_type"].lower()}_c{CLASS_IDX}_curves.png'
    plt.savefig(out, dpi=150); plt.show(); print(f'Saved to {out}')
else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(history['step'], history['init_dice_mean'],  label='Init Dice (U-Net)', ls='--', alpha=0.6)
    ax1.plot(history['step'], history['final_dice_mean'], label=f'Final Dice ({cfg["agent_type"]})')
    ax1.set(xlabel='Training step', ylabel='Mean Val Dice', title='Refinement learning curve')
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(history['step'], history['delta_dice_mean'], color='C2')
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set(xlabel='Training step', ylabel='Delta Dice (final - init)',
            title=f'Refinement gain ({cfg["agent_type"]} — {CLASS_NAME})')
    ax2.grid(alpha=0.3)
    plt.suptitle(f'CAMUS {cfg["agent_type"]} — {CLASS_NAME} learning curves')
    plt.tight_layout()
    out = f'/kaggle/working/{cfg["agent_type"].lower()}_c{CLASS_IDX}_curves.png'
    plt.savefig(out, dpi=150); plt.show(); print(f'Saved to {out}')
`;

const TEST = `
import json

if IS_TRACING:
    test_metrics = cv.evaluate_trace_testset(agent, test_samples, ENV_KW)
    print(json.dumps(test_metrics, indent=2))
    summary = {**test_metrics,
               'agent_type':   cfg['agent_type'],
               'target_class': cfg['target_class'],
               'class_name':   cfg['class_name'],
               'paradigm':     'contour_tracing'}
else:
    from iteris.drl_training import evaluate_agent
    test_metrics = evaluate_agent(agent, test_samples, env_kwargs=ENV_KW)
    print(json.dumps(test_metrics, indent=2))
    summary = {**test_metrics,
               'agent_type':    cfg['agent_type'],
               'target_class':  cfg['target_class'],
               'class_name':    cfg['class_name'],
               'reward_mode':   cfg['reward_mode'],
               'best_val_dice': float(best_dice)}

out = f'/kaggle/working/{cfg["agent_type"].lower()}_c{CLASS_IDX}_test_results.json'
with open(out, 'w') as f: json.dump(summary, f, indent=2)
print(f'\\nSaved to {out}')
`;

// ── Patch all three CAMUS notebooks ──────────────────────────────────────────

const NB = 'D:\\iteris\\notebooks\\';
const PATCHES = {
  'md-title':      updateMdTitle,
  'code-config':   updateAgentComment,
  'code-vizsetup': VIZSETUP,
  'code-compare':  COMPARE,
  'code-playback': PLAYBACK,
  'code-behaviour': BEHAVIOUR,
  'code-curves':   CURVES,
  'code-test':     TEST,
};

['03a_camus_drl_lv_endo.ipynb',
 '03b_camus_drl_lv_epi.ipynb',
 '03c_camus_drl_la.ipynb'].forEach(name => {
  patchNotebook(NB + name, PATCHES);
});

console.log('Done — CAMUS notebooks patched.');
