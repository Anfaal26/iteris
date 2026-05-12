"""
Patient-level train/val/test splitting.

Splitting at file level would leak views/phases of the same patient across
splits, inflating test metrics. Always split by patient ID.

Label-fraction subsampling (for few-shot ablations) is also patient-level —
keeping a fraction of training patients, not a fraction of training files.
"""

import random
from typing import List, Tuple


def patient_level_split(
    records: List[dict],
    val_split: float,
    test_split: float,
    label_frac: float,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Returns (train_records, val_records, test_records).

    Splits happen in this order:
      1. Shuffle patients with `seed`.
      2. Carve out test_split fraction → test_patients.
      3. Carve out val_split fraction → val_patients.
      4. Remainder is the full training pool.
      5. If `label_frac < 1.0`, keep only the first `label_frac` fraction
         of the training pool (after shuffle → deterministic with seed).

    Val and test always use the full data fraction — only training shrinks
    for few-shot experiments.
    """
    patients = sorted({r['patient'] for r in records})
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_test = max(1, int(len(patients) * test_split))
    n_val  = max(1, int(len(patients) * val_split))

    test_pts  = set(patients[:n_test])
    val_pts   = set(patients[n_test : n_test + n_val])
    train_pool = patients[n_test + n_val:]

    if label_frac < 1.0:
        n_keep = max(1, int(len(train_pool) * label_frac))
        train_pts = set(train_pool[:n_keep])
        print(f'[splits] Few-shot: keeping {n_keep}/{len(train_pool)} train patients '
              f'({label_frac*100:.0f}%)')
    else:
        train_pts = set(train_pool)

    train = [r for r in records if r['patient'] in train_pts]
    val   = [r for r in records if r['patient'] in val_pts]
    test  = [r for r in records if r['patient'] in test_pts]

    print(f'[splits] Patients — train: {len(train_pts)}  val: {len(val_pts)}  test: {len(test_pts)}')
    print(f'[splits] Samples  — train: {len(train)}  val: {len(val)}  test: {len(test)}')
    return train, val, test
