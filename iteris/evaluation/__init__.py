"""Standalone (re-)evaluation utilities for trained DRL checkpoints.

These are decoupled from the training loop so an already-trained checkpoint can
be re-scored on the test set — e.g. to correct a run whose reported numbers were
computed on the final-step agent instead of the deployable best-val checkpoint
(see re_eval_td3).
"""

from .re_eval_td3 import reeval_checkpoint

__all__ = ['reeval_checkpoint']
