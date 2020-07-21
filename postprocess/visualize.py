"""A copy from
https://github.com/DM-Berger/unet-learn/blob/6dc108a9a6f49c6d6a50cd29d30eac4f7275582e/src/lightning/log.py
"""

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
from matplotlib.pyplot import Axes, Figure
from matplotlib.text import Text
from numpy import ndarray
import numpy as np
from tqdm.auto import tqdm, trange
import os
import torch as t

from collections import OrderedDict
from numpy import ndarray
from matplotlib.pyplot import Axes, Figure
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from typing import Any, Dict, List, Tuple, Union
from pytorch_lightning.core.lightning import LightningModule

"""
For TensorBoard logging usage, see:
https://www.tensorflow.org/api_docs/python/tf/summary

For Lightning documentation / examples, see:
https://pytorch-lightning.readthedocs.io/en/latest/experiment_logging.html#tensorboard

NOTE: The Lightning documentation here is not obvious to newcomers. However,
`self.logger` returns the Torch TensorBoardLogger object (generally quite
useless) and `self.logger.experiment` returns the actual TensorFlow
SummaryWriter object (e.g. with all the methods you actually care about)

For the Lightning methods to access the TensorBoard .summary() features, see
https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loggers.html#pytorch_lightning.loggers.TensorBoardLogger

**kwargs for SummaryWriter constructor defined at
https://www.tensorflow.org/api_docs/python/tf/summary/create_file_writer
^^ these args look largely like things we don't care about ^^
"""


def make_imgs(img: ndarray, imin: Any = None, imax: Any = None) -> ndarray:
    """Apply a 3D binary mask to a 1-channel, 3D ndarray `img` by creating a 3-channel
    image with masked regions shown in transparent blue. """
    imin = img.min() if imin is None else imin
    imax = img.max() if imax is None else imax
    scaled = np.array(((img - imin) / (imax - imin)) * 255, dtype=int)  # img
    if len(img.shape) == 3:
        return scaled
    raise ValueError("Only accepts 1-channel or 3-channel images")


def get_logger(logdir: Path) -> TensorBoardLogger:
    return TensorBoardLogger(str(logdir), name="unet")


def slice_label(i: int, mids: Tensor, slicekey: str):
    quarts = mids // 2  # slices at first quarter of the way through
    quarts3_4 = mids + quarts  # slices 3/4 of the way through
    keymap = {"1/4": quarts, "1/2": mids, "3/4": quarts3_4}
    idx = keymap[slicekey]
    if i == 0:
        return f"[{idx[i]},:,:]"
    if i == 1:
        return f"[:,{idx[i]},:]"
    if i == 2:
        return f"[:,:,{idx[i]}]"

    f"[{idx[i]},:,:]", f"[:,{idx[i]},:]", f"[:,:,{idx[i]}]"
    raise IndexError("Only three dimensions supported.")


# https://www.tensorflow.org/tensorboard/image_summaries#logging_arbitrary_image_data
class BrainSlices:
    def __init__(self, lightning: LightningModule, img: Tensor, target_: Tensor, prediction: Tensor):
        # lol mypy type inference really breaks down here...
        self.lightning = lightning
        img_: ndarray = img.cpu().detach().numpy().squeeze()
        targ_: ndarray = target_.cpu().detach().numpy().squeeze()
        pred: ndarray = prediction.cpu().detach().numpy().squeeze()
        mids: ndarray = np.array(img_.shape) // 2
        quarts: ndarray = mids // 2  # slices at first quarter of the way through
        quarts3_4: ndarray = mids + quarts  # slices 3/4 of the way through
        self.mids = mids
        self.quarts = quarts
        self.quarts3_4 = quarts3_4
        self.slice_positions = ["1/4", "1/2", "3/4"]
        self.masknames = ["Actual Brain Tissue (probability)",
                          "Predicted Brain Tissue (probability)",
                          "Predicted Brain Tissue (binary)"]
        self.shape = np.array(img_.shape)

        # Use for mp4
        n_slices = 4
        self.orig = img_
        self.img = img_
        self.masks = [targ_, pred > 0.5]  # have threshold here, need to add
        slice_base = np.array(self.img.shape) // n_slices
        self.mask_video_names = ["Actual Brain Tissue (probability)",
                                 "Predicted Brain Tissue (binary)"]
        self.scale_imgs = make_imgs(self.img)

        self.imgs = OrderedDict(
            [
                ("1/4", (img_[quarts[0], :, :], img_[:, quarts[1], :], img_[:, :, quarts[2]])),
                ("1/2", (img_[mids[0], :, :], img_[:, mids[1], :], img_[:, :, mids[2]])),
                ("3/4", (img_[quarts3_4[0], :, :], img_[:, quarts3_4[1], :], img_[:, :, quarts3_4[2]])),
            ]
        )
        self.targets = OrderedDict(
            [
                ("1/4", (targ_[quarts[0], :, :], targ_[:, quarts[1], :], targ_[:, :, quarts[2]])),
                ("1/2", (targ_[mids[0], :, :], targ_[:, mids[1], :], targ_[:, :, mids[2]])),
                ("3/4", (targ_[quarts3_4[0], :, :], targ_[:, quarts3_4[1], :], targ_[:, :, quarts3_4[2]])),
            ]
        )
        self.preds = OrderedDict(
            [
                ("1/4", (pred[quarts[0], :, :], pred[:, quarts[1], :], pred[:, :, quarts[2]])),
                ("1/2", (pred[mids[0], :, :], pred[:, mids[1], :], pred[:, :, mids[2]])),
                ("3/4", (pred[quarts3_4[0], :, :], pred[:, quarts3_4[1], :], pred[:, :, quarts3_4[2]])),
            ]
        )
        self.labels = {
            "1/4": [f"[{quarts[0]},:,:]", f"[:,{quarts[1]},:]", f"[:,:,{quarts[2]}]"],
            "1/2": [f"[{mids[0]},:,:]", f"[:,{mids[1]},:]", f"[:,:,{mids[2]}]"],
            "3/4": [f"[{quarts3_4[0]},:,:]", f"[:,{quarts3_4[1]},:]", f"[:,:,{quarts3_4[2]}]"],
        }

    def plot(self) -> Tuple[Figure, Axes]:
        nrows, ncols = 3, 1  # one row for each slice position
        all_trues, all_targets, all_preds = [], [], []
        for i in range(3):  # We want this first so middle images are middle
            for j, position in enumerate(self.slice_positions):
                img, target = self.imgs[position][i], self.targets[position][i]
                prediction = self.preds[position][i]
                all_trues.append(img)
                all_targets.append(target)
                all_preds.append(prediction)
        fig: Figure
        axes: Axes
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)
        true = np.concatenate(all_trues, axis=1)
        target = np.concatenate(all_targets, axis=1)
        pred = np.concatenate(all_preds, axis=1)
        # convert to logits, since we are using BCEWithLogitsLoss. That is,
        # BDEWithLogitsLoss handles the sigmoid + loss internally to avoid
        # imprecision issues, but then this means the output of our network
        # never *actually* passes through a sigmoid. So we do that here.
        # pred = t.sigmoid(t.tensor(pred)).numpy()

        # Consistently apply colormap since images are standardized but still
        # vary considerably in maximum and minimum values
        imin = np.min(true)
        imax = np.max(true)
        scaled = np.array(((true - imin) / (imax - imin)) * 255, dtype=int)
        # The alpha blending value, between 0 (transparent) and 1 (opaque). This parameter is ignored for RGBA input data.
        true_args = dict(vmin=0, vmax=255, cmap="gray", alpha=0.5)
        mask_args = dict(vmin=0.0, vmax=1.0, cmap="gray", alpha=0.5)

        # Display data as an image; i.e. on a 2D regular raster.
        axes[0].imshow(scaled, **true_args)
        axes[0].imshow(target, **mask_args)
        axes[0].set_title(self.masknames[0])
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].imshow(scaled, **true_args)
        axes[1].imshow(pred, **mask_args)
        axes[1].set_title(self.masknames[1])
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        axes[2].imshow(scaled * np.array(pred > 0.5, dtype=float), **true_args)
        axes[2].set_title(self.masknames[2])
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        fig.tight_layout(h_pad=0)
        fig.subplots_adjust(hspace=0.0, wspace=0.0)
        return fig, axes

    def visualize(self, batch_idx: int, outdir: Path = None) -> None:
        fig, axes = self.plot()
        if self.lightning.show_plots:
            if outdir is None:  # for local debugging
                plt.show()
                plt.close()
                return
            fig.set_size_inches(w=10, h=6)
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(outdir / f"epoch{self.lightning.current_epoch}_batch{batch_idx}.png", dpi=200)
            plt.close()
            return

    def log(self, batch_idx: int, title: str, dice_score: float) -> None:
        logger = self.lightning.logger
        fig, axes = self.plot()
        summary = f"{title}: Epoch {self.lightning.current_epoch + 1}, Batch {batch_idx}, dice: {dice_score}"
        logger.experiment.add_figure(summary, fig, close=True)

        # if you want to manually intervene, look at the code at
        # https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/_utils.py
        # permalink to version:
        # https://github.com/pytorch/pytorch/blob/780fa2b4892512b82c8c0aaba472551bd0ce0fad/torch/utils/tensorboard/_utils.py#L5
        # then use logger.experiment.add_image(summary, image)

    # code is borrowed from: https://github.com/DM-Berger/autocrop/blob/master/autocrop/visualize.py#L125
    def animate_masks(
            self,
            dpi: int = 100,
            n_frames: int = 300,
            fig_title: str = None,
            outfile: Path = None,
    ) -> None:
        def get_slice(img: ndarray, mask: ndarray, ratio: float, threshold: float = 0.5) -> ndarray:
            """Returns eig_img, raw_img"""
            img = np.where(mask > threshold, 255, img)
            if ratio < 0 or ratio > 1:
                raise ValueError("Invalid slice position")
            if len(img.shape) == 3:
                x_max, y_max, z_max = np.array(img.shape, dtype=int)
                x, y, z = np.array(np.floor(np.array(img.shape) * ratio), dtype=int)
            elif len(img.shape) == 4:
                x_max, y_max, z_max, _ = np.array(img.shape, dtype=int)
                x, y, z = np.array(np.floor(np.array(img.shape[:-1]) * ratio), dtype=int)
            x = int(10 + ratio * (x_max - 20))  # make x go from 10:-10 of x_max
            y = int(10 + ratio * (y_max - 20))  # make x go from 10:-10 of x_max
            x = x - 1 if x == x_max else x
            y = y - 1 if y == y_max else y
            z = z - 1 if z == z_max else z
            return np.concatenate([img[x, :, :], img[:, y, :], img[:, :, z]], axis=1)

        def init_frame(img: ndarray, mask: ndarray, ratio: float, fig: Figure, ax: Axes, title) -> Tuple[
            AxesImage, Colorbar, Text]:
            image_slice = get_slice(img, mask=mask, ratio=ratio)
            true_args = dict(vmin=0, vmax=255, cmap="gray", alpha=0.7)

            im = ax.imshow(image_slice, animated=True, **true_args)
            # im = ax.imshow(image_slice, animated=True)
            ax.set_xticks([])
            ax.set_yticks([])
            title = ax.set_title(title)
            cb = fig.colorbar(im, ax=ax)
            print()
            return im, cb, title

        def update_axis(img: ndarray, mask: ndarray, ratio: float, im: AxesImage) -> AxesImage:
            image_slice = get_slice(img, mask=mask, ratio=ratio)
            # vn, vm = get_vranges()
            im.set_data(image_slice)
            # im.set_clim(vn, vm)
            # we don't have to update cb, it is linked
            return im

        # owe a lot to below for animating the colorbars
        # https://stackoverflow.com/questions/39472017/how-to-animate-the-colorbar-in-matplotlib
        def init() -> Tuple[Figure, Axes, List[AxesImage], List[Colorbar]]:
            fig: Figure
            axes: Axes
            fig, axes = plt.subplots(nrows=len(self.masks), ncols=1, sharex=False, sharey=False)  # 3

            ims: List[AxesImage] = []
            cbs: List[Colorbar] = []

            for ax, img, mask, title in zip(axes.flat, self.scale_imgs, self.masks, self.mask_video_names):
                im, cb, title = init_frame(img=img, mask=mask, ratio=0.0, fig=fig, ax=ax, title=title)
                ims.append(im)
                cbs.append(cb)

            if fig_title is not None:
                fig.suptitle(fig_title)
            # fig.tight_layout(h_pad=0)
            fig.set_size_inches(w=10, h=5)
            fig.subplots_adjust(hspace=0.2, wspace=0.0)
            plt.savefig("./first_img.jpg")
            return fig, axes, ims, cbs

        # def show_init_flg():
        #     fig, axes = plt.subplots()


        N_FRAMES = n_frames
        ratios = np.linspace(0, 1, num=N_FRAMES)

        fig, axes, ims, cbs = init()


        # awkward, but we need this defined after to close over the above variables
        def animate(f: int) -> Any:
            ratio = ratios[f]
            updated = []
            for im, img, mask in zip(ims, self.scale_imgs, self.masks):
                updated.append(update_axis(img=img, mask=mask, ratio=ratio, im=im))
            return updated

        ani = animation.FuncAnimation(
            fig=fig,
            func=animate,
            frames=N_FRAMES,
            blit=False,
            interval=3600 / N_FRAMES,
            repeat_delay=100 if outfile is None else None,
        )

        if outfile is None:
            plt.show()
        else:
            pbar = tqdm(total=100, position=1, desc='mp4')
            def prog_logger(current_frame: int, total_frames: int = N_FRAMES) -> Any:
                if (current_frame % (total_frames // 10)) == 0 and (current_frame != 0):
                    pbar.update(10)
                # tqdm.write("Done task %i" % (100 * current_frame / total_frames))
                #     print("Saving... {:2.1f}%".format(100 * current_frame / total_frames))

            # writervideo = animation.FFMpegWriter(fps=60)
            ani.save(outfile, codec="h264", dpi=dpi, progress_callback=prog_logger)
            # ani.save(outfile, progress_callback=prog_logger, writer=writervideo)
            pbar.close()


def log_weights(module: LightningModule) -> None:
    for name, param in module.named_parameters():
        module.logger.experiment.add_histogram(name, param, global_step=module.global_step)


"""
Actual methods on logger.experiment can be found here!!!
https://pytorch.org/docs/stable/tensorboard.html
"""


def log_all_info(module: LightningModule, img: Tensor, target: Tensor, logist: Tensor, batch_idx: int,
                 title: str, dice_score: float) -> None:
    """Helper for decluttering training loop. Just performs all logging functions."""
    brainSlice = BrainSlices(module, img, target, logist)
    brainSlice.log(batch_idx, title, dice_score)

    if not os.path.exists('./mp4'):
        os.mkdir('./mp4')

    brainSlice.animate_masks(fig_title=f"epoch: {module.current_epoch}, batch: {batch_idx}, dice_score: {dice_score}",
                             outfile=Path(
                                 f"./mp4/epoch={module.current_epoch}_batch={batch_idx}_dice_score={dice_score}.mp4"))
    log_weights(module)
