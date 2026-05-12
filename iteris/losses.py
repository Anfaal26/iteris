"""Loss functions."""

from monai.losses import DiceCELoss


def build_loss(cfg: dict):
    """
    Combined Dice + CrossEntropy loss, background excluded.
    Standard for medical segmentation across all our datasets.
    """
    alpha = cfg.get('loss_alpha', 0.5)
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=alpha,
        lambda_ce=1.0 - alpha,
    )
