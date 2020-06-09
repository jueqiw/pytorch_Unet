import numpy as np
import matplotlib.pyplot as plt
import torch as t

from matplotlib.pyplot import Axes, Figure
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor


def visualize(img: Tensor, target: Tensor, prediction: Tensor) -> None:
    def slice_label(i: int, mids: Tensor):
        if i == 0:
            return f"[{mids[i]},:,:]"
        if i == 1:
            return f"[:,{mids[i]},:]"
        if i == 2:
            return f"[:,:,{mids[i]}]"
        raise IndexError("Only three dimensions supported.")

    img = img.cpu().detach().numpy().squeeze()
    target = target.cpu().detach().numpy().squeeze()
    pred = prediction.cpu().detach().numpy().squeeze()
    mids = np.array(img.shape) // 2
    img_slices = [img[mids[0], :, :], img[:, mids[1], :], img[:, :, mids[2]]]
    target_slices = [target[mids[0], :, :], target[:, mids[1], :], target[:, :, mids[2]]]
    pred_slices = [pred[mids[0], :, :], pred[:, mids[1], :], pred[:, :, mids[2]]]

    fig: Figure
    axes: Axes
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    for i, (img, target, pred) in enumerate(zip(img_slices, target_slices, pred_slices)):
        ax, pred_ax = axes[0][i], axes[1][i]
        # plot true img and true segmentations
        ax.imshow(img, cmap="inferno", label="img")
        ax.imshow(target, cmap="inferno", label="mask", alpha=0.5)
        ax.set_title("True Segmentation")
        ax.set_xlabel(slice_label(i, mids))

        # plot predicted segmentations on true img
        pred_ax.imshow(img, cmap="inferno", label="img")
        pred_ax.imshow(pred, cmap="inferno", label="pred", alpha=0.5)
        pred_ax.set_xlabel(slice_label(i, mids))
        pred_ax.set_title("Predicted Segmentation")
    plt.show()