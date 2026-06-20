"""
Validate that TD3 (robust DDPG) on the contour env LEARNS to beat the U-Net
baseline -- the property a 3-D global DDPG structurally cannot have.

Synthetic disks (filled circles); init = GT eroded -> uniformly under-segmented,
so pushing contour sectors outward strictly raises Dice, then holding keeps it.
A correctly-wired TD3 agent learns positive sector displacements where the mask
is too small and ~zero elsewhere, ending ABOVE baseline. Two regimes:

  EASY : erode 2px, small disks  -> baseline ~0.86 (large headroom, sanity).
  HARD : erode 1px, large disks  -> baseline ~0.95 (tiny headroom, the real
         CAMUS/BRISC-like regime where the reward signal is small).

ASCII-only prints (Windows cp1252 safe). Run:
  set PYTHONIOENCODING=utf-8
  F:\\Anaconda\\envs\\fyp_env\\python.exe D:\\iteris\\scripts\\validate_td3_contour.py
"""
import sys, time
import numpy as np
import scipy.ndimage as ndi

sys.path.insert(0, 'D:/iteris')
from iteris.drl_training import run_drl_training, evaluate_agent, _build_state_caches
from iteris.env_contour_refine import ContourRefineEnv
from iteris.env import dice_score

H = 64


def make_sample(seed, erode_iters, rad_lo, rad_hi):
    r = np.random.RandomState(seed)
    cy = r.randint(rad_hi + 3, H - rad_hi - 3)
    cx = r.randint(rad_hi + 3, H - rad_hi - 3)
    rad = r.randint(rad_lo, rad_hi)
    yy, xx = np.mgrid[0:H, 0:H]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    gt = (dist <= rad).astype(np.uint8)
    init = ndi.binary_erosion(gt, iterations=erode_iters).astype(np.uint8)
    if not init.any():
        init = gt.copy()
    prob = 1.0 / (1.0 + np.exp((dist - rad) / 1.5))
    image = gt.astype(np.float32) * 0.5 + r.randn(H, H).astype(np.float32) * 0.05
    return dict(image=image.astype(np.float32), gt_mask=gt, init_mask=init,
                prob_map=prob.astype(np.float16), patient=f'syn{seed}', view='', phase='')


def regime(erode, lo, hi):
    tr = [make_sample(s, erode, lo, hi) for s in range(32)]
    va = [make_sample(s, erode, lo, hi) for s in range(1000, 1008)]
    return tr, va


def run(tag, tr, va, steps=2500):
    base = float(np.mean([dice_score(s['init_mask'], s['gt_mask']) for s in va]))
    cfg = dict(
        agent_type='TD3', env_class='contour', image_size=H, target_class=1, dataset='syn',
        reward_mode='dice_pbrs', reward_potential_scale=12.0,
        n_points=24, cont_sectors=12, disp_px=1.5, spline_smooth=1.5,
        max_steps=8, sdt_clip=20.0, reward_clip=2.0, disable_auto_stop=False,
        hard_mining_scale=2.0,
        actor_lr=2e-4, critic_lr=1e-3, gamma=0.99, tau=0.01,
        policy_delay=2, expl_noise=0.2, target_noise=0.2, noise_clip=0.5,
        embed_dim=64, batch_size=64, buffer_size=6000, prefill_steps=800,
        train_steps=steps, eval_every=steps, seed=0,
        checkpoint_dir='D:/iteris/scripts/_ckpt',
    )
    t0 = time.time()
    out = run_drl_training(cfg, tr, va)
    env_kwargs = dict(action_type='continuous', reward_mode='dice_pbrs',
                      reward_potential_scale=12.0, n_points=24, cont_sectors=12,
                      disp_px=1.5, spline_smooth=1.5, max_steps=8, sdt_clip=20.0,
                      disable_auto_stop=False, pbrs_gamma=0.99)
    caches = _build_state_caches(va, H)
    m = evaluate_agent(out['agent'], caches, env_kwargs, env_cls=ContourRefineEnv)
    delta = m['final_dice_mean'] - m['init_dice_mean']
    verdict = 'PASS' if delta > 0.005 else ('TIE ' if delta > -0.002 else 'FAIL')
    print(f"  [{verdict}] {tag:5s} TD3-contour | base {base:.4f} "
          f"init {m['init_dice_mean']:.4f} -> final {m['final_dice_mean']:.4f} "
          f"(delta {delta:+.4f}, best-seen {m['best_dice_mean']:.4f}) [{time.time()-t0:.0f}s]", flush=True)
    return delta


print('=== TD3-on-contour validation (robust DDPG, angular-sector action) ===', flush=True)
print('\n[EASY] erode 2px small disks, baseline ~0.86', flush=True)
d_easy = run('EASY', *regime(2, 11, 17))
print('\n[HARD] erode 1px large disks, baseline ~0.95', flush=True)
d_hard = run('HARD', *regime(1, 22, 28))
print('\n=== SUMMARY ===', flush=True)
print(f"EASY delta {d_easy:+.4f} | HARD delta {d_hard:+.4f} -> "
      f"{'TD3 BEATS BASELINE' if (d_easy > 0.005 and d_hard > 0.005) else 'NEEDS TUNING'}", flush=True)
