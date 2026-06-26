"""Loss functions."""

from monai.losses import DiceCELoss


def build_loss(cfg: dict):
    """
    Combined Dice + CrossEntropy loss, background excluded.
    Standard for medical segmentation across all our datasets.

    `loss_label_smoothing` (default 0.0, opt-in via cfg) softens the CE term's
    targets so the trained net's softmax outputs aren't pinned at 0/1 — needed
    for any model whose output prob_map feeds the DRL uncertainty gate or the
    contour env's 5th state channel (see iteris/diagnostics.py::pillar4_report,
    `prob_map_informativeness`): an overconfident net makes that channel near-
    binary/inert. Only changes the CE term; the Dice term's smooth_nr/smooth_dr
    are unrelated numerical-stability epsilons, not label smoothing.
    """
    alpha = cfg.get('loss_alpha', 0.5)
    label_smoothing = cfg.get('loss_label_smoothing', 0.0)
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=alpha,
        lambda_ce=1.0 - alpha,
        label_smoothing=label_smoothing,
    )
