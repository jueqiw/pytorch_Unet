import torch

CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4


def get_dice_score(output, target, epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    iou_score = (tp + epsilon) / (tp + fp + fn)
    return dice_score, iou_score


def get_dice_loss(output, target):
    dice_score, iou = get_dice_loss(output, target)
    return 1 - dice_score
