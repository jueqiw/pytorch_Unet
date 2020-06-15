import pytorch_lightning as pl
from torchio import DATA
from typing import Optional, Union, List
from argparse import ArgumentParser
from torch.nn import MultiLabelSoftMarginLoss
from torch.utils.data import DataLoader
from data.get_subjects import get_subjects
from data.const import CC359_DATASET_DIR, NFBS_DATASET_DIR, ADNI_DATASET_DIR_1, COMPUTECANADA
from data.transform import get_train_transforms, get_val_transform
from data.const import SIZE
from unet.unet import *
from utils.loss import get_dice_score
import torch.nn.functional as F
from postprocess.visualize import BrainSlices
from pathlib import Path
import multiprocessing
from torch import Tensor
import torchio
import torch


class LitUnet(pl.LightningModule):

    def __init__(
            self,
            hparams
    ):
        super().__init__()
        self.hparams = hparams
        self.unet = UNet(
            in_channels=1,
            out_classes=1,
            num_encoding_blocks=3,
            out_channels_first_layer=8,
            normalization=self.hparams.normalization,
            upsampling_type='conv',
            padding=2,
            activation='PReLU',
        )
        if COMPUTECANADA:
            datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR, ADNI_DATASET_DIR_1]
        else:
            datasets = [CC359_DATASET_DIR]
        subjects = get_subjects(datasets)
        num_subjects = len(subjects)
        num_training_subjects = int(num_subjects * 0.9)  # （5074+359+21） * 0.9 used for training
        self.training_subjects = subjects[:num_training_subjects]
        self.validation_subjects = subjects[num_training_subjects:]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10000, help='Number of epochs',
                            dest='epochs')
        parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=1, help='Batch size',
                            dest='batch_size')
        parser.add_argument('-l', '--learning_rate', metavar='LR', type=float, nargs='?', default=1e-3,
                            help='Learning rate')
        parser.add_argument('-n', '--normalization', metavar='E', type=str, default="Group", help='the way of '
                                                                                                  'normalization')
        parser.add_argument('-d', '--down_sample', type=str, default="max", help="the way to down sample")
        parser.add_argument('--loss', type=str, nargs='?', default="BCEWL", help='Loss Function')
        # parser.add_argument('-r', '--run', dest='run', type=int, default=1, help='run times')
        parser.add_argument('-p', '--show_plot', type=bool, default=False, help='whether to plot the figure')
        return parser

    def forward(self, x: Tensor) -> Tensor:
        return self.unet(x)

    def train_dataloader(self) -> DataLoader:
        training_transform = get_train_transforms()
        train_imageDataset = torchio.ImagesDataset(self.training_subjects, transform=training_transform)
        training_loader = DataLoader(train_imageDataset, batch_size=self.hparams.batch_size,
                                     num_workers=multiprocessing.cpu_count())
        print('Training set:', len(train_imageDataset), 'subjects')
        return training_loader

    def val_dataloader(self) -> DataLoader:
        val_transform = get_val_transform()
        val_imageDataset = torchio.ImagesDataset(self.validation_subjects, transform=val_transform)
        val_loader = DataLoader(val_imageDataset, batch_size=self.hparams.batch_size * 2,
                                num_workers=multiprocessing.cpu_count())
        print('Validation set:', len(val_imageDataset), 'subjects')
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)  # ??
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, targets = batch["img"][DATA], batch["label"][DATA]
        logits = self(inputs)
        prob = torch.sigmoid(logits)
        dice, iou = get_dice_score(prob, targets)
        if int(batch_idx) != 0 and self.hparams.show_plot and int(batch_idx) % 25 == 0:
            slices = BrainSlices(inputs, targets, logits)
            slices.visualize(int(batch_idx), self.current_epoch,
                             outdir=Path(__file__).resolve().parent / "log" / "plot")
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        tensorboard_logs = {"train_loss": loss, "train_IoU": iou.mean(), "train_dice": dice.mean()}
        return {'loss': loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_id):
        inputs, targets = batch["img"][DATA], batch["label"][DATA]
        logits = self(inputs)
        prob = torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        dice, iou = get_dice_score(prob, targets)
        tensorboard_logs = {"val_loss": loss, "val_IoU": iou.mean(), "val_dice": dice.mean()}
        return {'val_loss': loss, 'val_step_IoU': iou.mean(), 'log': tensorboard_logs}

    # Called at the end of the validation epoch with the outputs of all validation steps.
    def validation_epoch_end(self, outputs):
        # torch.stack: Concatenates sequence of tensors along a new dimension.
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_IoU = torch.stack([x['val_step_IoU'] for x in outputs]).mean()
        return {'val_loss': avg_loss, 'val_IoU': avg_val_IoU}
