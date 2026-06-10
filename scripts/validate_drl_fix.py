"""
Controlled validation that the DRL fix lets DQN / DuelingDDQN BEAT baseline.

We cannot run Kaggle CAMUS/BRISC locally, so this isolates the LEARNING
MACHINERY on synthetic tasks where improvement over the "U-Net" baseline is
genuinely achievable through the available global actions.

Two regimes:
  EASY  : init = GT eroded by 2 px  -> baseline Dice ~0.86, large headroom.
          Sanity check that the pipeline trains end-to-end at all.
  HARD  : init = GT eroded by 1 px on LARGE disks -> baseline Dice ~0.95,
          per-step headroom ~0.01-0.03. This is the regime that exposes the
          discount-drag failure: with an UN-centred potential Phi=Dice~0.95 and
          gamma=0.99, holding pays (gamma-1)*Phi ~ -0.0095/step and small real
          gains net negative, so the agent collapses to STOP-at-baseline.
          With the baseline-centred, scaled potential the agent should instead
          dilate a step or two and STOP ABOVE baseline.

ASCII-only prints (Windows cp1252 console safe). Run:
  set PYTHONIOENCODING=utf-8
  F:\\Anaconda\\envs\\fyp_env\\python.exe D:\\iteris\\scripts\\validate_drl_fix.py
"""
import sys, time
import numpy as np
import scipy.ndimage as ndi

sys.path.insert(0, 'D:/iteris')
from iteris.drl_training import run_drl_training, evaluate_agent, _build_state_caches
from iteris.env import SegmentationEnv, dice_score

H = 64


def make_sample(seed, erode_iters, rad_lo, rad_hi):
    r = np.random.RandomState(seed)
    cy, cx = r.randint(rad_hi + 2, H - rad_hi - 2), r.randint(rad_hi + 2, H - rad_hi - 2)
    rad = r.randint(rad_lo, rad_hi)
    yy, xx = np.mgrid[0:H, 0:H]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    gt = (dist <= rad).astype(np.uint8)
    init = ndi.binary_erosion(gt, iterations=erode_iters).astype(np.uint8)
    if not init.any():
        init = gt.copy()
    # soft prob map: high inside GT, ramps across the boundary
    prob = 1.0 / (1.0 + np.exp((dist - rad) / 1.5))
    image = gt.astype(np.float32) * 0.5 + r.randn(H, H).astype(np.float32) * 0.05
    return dict(image=image.astype(np.float32), gt_mask=gt, init_mask=init,
                prob_map=prob.astype(np.float16), patient=f'syn{seed}', view='', phase='')


def build_regime(erode_iters, rad_lo, rad_hi):
    train = [make_sample(s, erode_iters, rad_lo, rad_hi) for s in range(40)]
    val = [make_sample(s, erode_iters, rad_lo, rad_hi) for s in range(1000, 1016)]
    return train, val


def run(tag, agent_type, train_samples, val_samples, steps=4000, K=10.0):
    base = np.mean([dice_score(s['init_mask'], s['gt_mask']) for s in val_samples])
    cfg = dict(
        agent_type=agent_type, image_size=H, target_class=1, dataset='syn',
        reward_mode='dice_pbrs', reward_potential_scale=K,
        max_steps=12, shift_px=1, sdt_clip=20.0, reward_clip=2.0,
        disable_auto_stop=True, reward_step_penalty=0.0, hard_mining_scale=2.0,
        lr=3e-4, gamma=0.99, tau=0.01, embed_dim=128,
        batch_size=64, buffer_size=8000, prefill_steps=1500,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=int(steps * 0.6),
        train_steps=steps, eval_every=steps,
        checkpoint_dir='D:/iteris/scripts/_ckpt', seed=0,
    )
    t0 = time.time()
    out = run_drl_training(cfg, train_samples, val_samples)
    env_kwargs = dict(action_type='discrete', reward_mode='dice_pbrs',
                      reward_potential_scale=K, max_steps=12, shift_px=1,
                      sdt_clip=20.0, disable_auto_stop=True, pbrs_gamma=0.99)
    caches = _build_state_caches(val_samples, H)
    m = evaluate_agent(out['agent'], caches, env_kwargs, env_cls=SegmentationEnv)
    delta = m['final_dice_mean'] - m['init_dice_mean']
    verdict = 'PASS' if delta > 0.005 else ('TIE ' if delta > -0.002 else 'FAIL')
    print(f"  [{verdict}] {tag:5s} {agent_type:8s} | base {base:.4f} "
          f"init {m['init_dice_mean']:.4f} -> final {m['final_dice_mean']:.4f} "
          f"(delta {delta:+.4f}, best-seen {m['best_dice_mean']:.4f}) [{time.time()-t0:.0f}s]")
    return delta


print('=== DRL fix validation (baseline-centred scaled PBRS) ===', flush=True)

print('\n[EASY regime] init = GT eroded 2px, baseline ~0.86', flush=True)
tr_e, va_e = build_regime(erode_iters=2, rad_lo=11, rad_hi=18)
run('EASY', 'DQN', tr_e, va_e)

print('\n[HARD regime] init = GT eroded 1px on large disks, baseline ~0.95', flush=True)
tr_h, va_h = build_regime(erode_iters=1, rad_lo=22, rad_hi=28)
d_dqn = run('HARD', 'DQN', tr_h, va_h)
d_due = run('HARD', 'DUELING', tr_h, va_h)

print('\n=== SUMMARY ===', flush=True)
ok = (d_dqn > 0.005) and (d_due > 0.005)
print(f"HARD-regime DQN delta {d_dqn:+.4f} | DUELING delta {d_due:+.4f} -> "
      f"{'BOTH BEAT BASELINE' if ok else 'STILL NOT BEATING BASELINE'}", flush=True)
