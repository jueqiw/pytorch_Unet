import pytorch_lightning as pl
from torchio import DATA
from torch.utils.data import DataLoader
from data.get_subjects import get_subjects
from data.const import CC359_DATASET_DIR, NFBS_DATASET_DIR, ADNI_DATASET_DIR_1, COMPUTECANADA
from data.transform import get_train_transforms, get_val_transform
from argparse import ArgumentParser
from data.const import SIZE
from model.unet.unet import UNet
from utils.loss import get_dice_score
import torch.nn.functional as F
from postprocess.visualize import log_all_info
import multiprocessing
from torch import Tensor
import torchio
import torch


class Lightning_Unet(pl.LightningModule):
    def __init__(self, hparams):
        super(Lightning_Unet, self).__init__()
        self.hparams = hparams
        self.unet = UNet(
            in_channels=1,
            out_classes=1,
            num_encoding_blocks=3,
            out_channels_first_layer=8,
            normalization=hparams.normalization,
            upsampling_type='conv',
            padding=2,
            dropout=0,
            all_size_input=hparams.all_size_input,
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
        # self.training_subjects = subjects[:10]
        # self.validation_subjects = subjects[10:15]

    def forward(self, x: Tensor) -> Tensor:
        return self.unet(x)

    def train_dataloader(self) -> DataLoader:
        training_transform = get_train_transforms()
        train_imageDataset = torchio.ImagesDataset(self.training_subjects, transform=training_transform)
        training_loader = DataLoader(train_imageDataset,
                                     batch_size=self.hparams.batch_size,
                                     # batch_size=1,
                                     num_workers=multiprocessing.cpu_count())
        print('Training set:', len(train_imageDataset), 'subjects')
        return training_loader

    def val_dataloader(self) -> DataLoader:
        val_transform = get_val_transform()
        val_imageDataset = torchio.ImagesDataset(self.validation_subjects, transform=val_transform)
        val_loader = DataLoader(val_imageDataset,
                                batch_size=1,  # always one because using different img size
                                # batch_size=2,
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
        if int(batch_idx) == 5:  # every epoch only save one fig
            log_all_info(self, inputs, targets, logits, batch_idx)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        tensorboard_logs = {"train_loss": loss, "train_IoU": iou, "train_dice": dice}
        return {'loss': loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_id):
        inputs, targets = batch["img"][DATA], batch["label"][DATA]
        # print(f"inputs shape: {inputs.shape}")
        if not self.hparams.all_size_input:
            inputs = F.interpolate(batch["img"][DATA], size=(SIZE, SIZE, SIZE))
            logits = self(inputs)
            shape = targets.shape
            prob = torch.sigmoid(logits)
            logits = F.interpolate(logits, size=(shape[2], shape[3], shape[4]))
            prob = F.interpolate(prob, size=(shape[2], shape[3], shape[4]))
        else:
            logits = self(inputs)
            prob = torch.sigmoid(logits)
        # print(f"prob shape: {prob.shape}")
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        dice, iou = get_dice_score(prob, targets)
        # tensorboard_logs = {"val_loss": loss, "val_IoU": iou, "val_dice": dice}
        if iou.mean() > 1.0:  # need to fix the initial part
            print(f"prob max: {torch.max(prob)}")
            print(f"target max: {torch.max(targets)}")
            raise Exception("val_IoU > 1 ???")
        return {'val_step_loss': loss, 'val_step_IoU': iou, 'val_step_dice': dice}

    # Called at the end of the validation epoch with the outputs of all validation steps.
    def validation_epoch_end(self, outputs):
        # torch.stack: Concatenates sequence of tensors along a new dimension.
        avg_loss = torch.stack([x['val_step_loss'] for x in outputs]).mean()
        avg_val_IoU = torch.stack([x['val_step_IoU'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_IoU': avg_val_IoU}
        return {'val_loss': avg_loss, 'val_IoU': avg_val_IoU, "log": tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        parameters defined here will be available to the model through self.hparams
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=1, help='Batch size', dest='batch_size')
        parser.add_argument("--learning_rate", type=float, default=1e-3, help='Learning rate')
        parser.add_argument("--normalization", type=str, default='Group', help='the way of normalization')
        parser.add_argument("--down_sample", type=str, default="max", help="the way to down sample")
        parser.add_argument("--loss", type=str, default="BCEWL", help='Loss Function')
        parser.add_argument("--run", dest='run', type=int, default=1, help='run times')
        parser.add_argument("--all_size_input", dest='all_size_input', type=bool, default=False, help='whether the '
                                                                                                      'input is not '
                                                                                                      'resize')
        return parser
