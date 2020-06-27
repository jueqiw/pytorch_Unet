import torch

CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4


def get_score(output, target, epsilon=1e-9):
    tp = ((output > 0.5) * (target > 0.5)).to(torch.long).sum(SPATIAL_DIMENSIONS)
    fp = ((output > 0.5) * (target < 0.5)).to(torch.long).sum(SPATIAL_DIMENSIONS)
    tn = ((output < 0.5) * (target < 0.5)).to(torch.long).sum(SPATIAL_DIMENSIONS)
    fn = ((output < 0.5) * (target > 0.5)).to(torch.long).sum(SPATIAL_DIMENSIONS)
    sup = (target == 1).to(torch.long).sum(SPATIAL_DIMENSIONS)

    dice_score = (2*tp + epsilon) / (2*tp + fp + fn)
    iou_score = (tp + epsilon) / (tp + fp + fn)
    sensitivity_score = (tp + epsilon) / (tp + fn)
    specificity_score = (tn + epsilon) / (tn + fp)
    return torch.mean(dice_score), torch.mean(iou_score), torch.mean(sensitivity_score), torch.mean(specificity_score)


def dice_loss(prob, target):
    """
    code is from https://github.com/CBICA/Deep-BET/blob/master/Deep_BET/utils/losses.py#L11
    :param input:
    :param target:
    :return:
    """
    smooth = 1e-7
    iflat = prob.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))