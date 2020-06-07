import torch
import numpy as np

CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

def matrix(probs, targets):
    SMOOTH = 1e-6
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    _iou = []
    _dice = []
    for prob, target in zip(probs, targets):
        threshold = torch.ones_like(prob).to(device) * 0.5

        # great than 0.5
        pred = torch.gt(prob, threshold)
        mask_bool = torch.gt(target.float(), threshold)

        _and = (pred & mask_bool).float().sum()
        _or = (pred | mask_bool).float().sum()
        iou = ((_and + SMOOTH) / (_or + SMOOTH)).sum()

        pred_sum = pred.float().sum()
        mask_bool_sum = mask_bool.float().sum()
        dice = ((2 * _and) / (pred_sum + mask_bool_sum)).sum()

        print(iou.shape)
        _iou.append(iou)
        _dice.append(dice)

    _iou = np.array(_iou)
    _dice = np.array(_dice)
    return _iou.mean(), _dice.mean()
