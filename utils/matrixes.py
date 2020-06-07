import torch

CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

def matrix(prob, target):
    SMOOTH = 1e-6
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    threshold = torch.ones_like(prob).to(device) * 0.5
    # great than 0.5
    pred = torch.gt(prob, threshold)
    mask_bool = torch.gt(target.float(), threshold)

    _and = (pred & mask_bool).float().sum(dim=SPATIAL_DIMENSIONS)
    _or = (pred | mask_bool).float().sum(dim=SPATIAL_DIMENSIONS)

    iou = ((_and + SMOOTH) / (_or + SMOOTH)).sum()

    pred_sum = pred.float().sum(dim=SPATIAL_DIMENSIONS)
    mask_bool_sum = mask_bool.float().sum(dim=SPATIAL_DIMENSIONS)

    dice = ((2 * _and) / (pred_sum + mask_bool_sum)).sum()
    return iou, dice, 1 - dice
