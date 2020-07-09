from typing import Union, List
import pytorch_lightning as pl
from torchio import DATA
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from data.get_subjects import get_subjects
from data.const import CC359_DATASET_DIR, NFBS_DATASET_DIR, ADNI_DATASET_DIR_1, COMPUTECANADA
from data.transform import get_train_transforms, get_val_transform, get_test_transform
from argparse import ArgumentParser
from data.const import SIZE
from model.unet.unet import UNet
from utils.loss import get_score, dice_loss
from utils.optimizer import fetch_optimizer
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from postprocess.visualize import log_all_info
from torch import Tensor
import torchio
import torch
import random


class Lightning_Unet(pl.LightningModule):
    def __init__(self, hparams, trial):
        super(Lightning_Unet, self).__init__()
        self.hparams = hparams

        # optimize
        self.lr = trial.suggest_loguniform("learning_rate", 1e-5, 1)
        self.weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1)
        self.normalization = trial.suggest_categorical('normalization', ['GroupNorm', 'InstanceNorm3d'])
        self.downsampling_type = trial.suggest_categorical('downsampling', ['conv', 'max'])
        self.kernal_size = trial.suggest_int('kernal_size', 3, 5)
        self.module_type = trial.suggest_categorical('module_type', ['Unet', 'ResUnet'])

        self.unet = UNet(
            in_channels=1,
            out_classes=1,
            num_encoding_blocks=3,
            out_channels_first_layer=8,
            kernal_size=self.kernal_size,
            normalization=self.normalization,
            module_type=self.module_type,
            downsampling_type=self.downsampling_type,
            padding=2,
            dropout=0,
        )
        # self.lr = self.hparams.learning_rate

        # self.max_loss_img = {
        #     "loss": 0,
        #     "img": None,
        #     "label": None,
        #     "prob": None,
        #     "batch_idx": None,
        #     "filename": None
        # }
        # self.min_loss_img = {
        #     "loss": 1,
        #     "img": None,
        #     "label": None,
        #     "prob": None,
        #     "batch_idx": None,
        #     "filename": None
        # }
        # self.subjects = get_subjects()
        # random.seed(42)
        # random.shuffle(self.subjects)  # shuffle it to pick the val set
        # if COMPUTECANADA:
        #     self.subjects = self.subjects[:500]
        # num_subjects = len(self.subjects)
        # num_training_subjects = int(num_subjects * 0.9)  # （5074+359+21） * 0.9 used for training
        # self.training_subjects = self.subjects[:num_training_subjects]
        # self.validation_subjects = self.subjects[num_training_subjects:]

    def forward(self, x: Tensor) -> Tensor:
        return self.unet(x)

    def setup(self, stage):
        self.subjects = get_subjects()
        random.seed(42)
        random.shuffle(self.subjects)  # shuffle it to pick the val set
        if COMPUTECANADA:
            self.subjects = self.subjects[:500]
        num_subjects = len(self.subjects)
        num_training_subjects = int(num_subjects * 0.9)  # （5074+359+21） * 0.9 used for training
        self.training_subjects = self.subjects[:num_training_subjects]
        self.validation_subjects = self.subjects[num_training_subjects:]

    def train_dataloader(self) -> DataLoader:
        training_transform = get_train_transforms()
        train_imageDataset = torchio.ImagesDataset(self.training_subjects, transform=training_transform)
        training_loader = DataLoader(train_imageDataset,
                                     batch_size=self.hparams.batch_size,
                                     # batch_size=1,
                                     # num_workers=multiprocessing.cpu_count()) would cause RuntimeError('DataLoader
                                     # worker (pid(s) {}) exited unexpectedly' if don't do that
                                     num_workers=4)
        print('Training set:', len(train_imageDataset), 'subjects')
        return training_loader

    def val_dataloader(self) -> DataLoader:
        val_transform = get_val_transform()
        val_imageDataset = torchio.ImagesDataset(self.validation_subjects, transform=val_transform)
        val_loader = DataLoader(val_imageDataset,
                                batch_size=self.hparams.batch_size * 2,
                                # num_workers=multiprocessing.cpu_count())
                                num_workers=4)
        print('Validation set:', len(val_imageDataset), 'subjects')
        return val_loader

    def test_dataloader(self):
        test_transform = get_test_transform()
        # using all the data to test
        test_imageDataset = torchio.ImagesDataset(self.subjects, transform=test_transform)
        test_loader = DataLoader(test_imageDataset,
                                 batch_size=1,  # always one because using different label size
                                 num_workers=4)
        print('Testing set:', len(test_imageDataset), 'subjects')
        return test_loader

    # need to adding more things
    def configure_optimizers(self):
        # Setting up the optimizer
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.lr,
                                    momentum=0.1,
                                    weight_decay=self.weight_decay)
                                    # weight_decay=0.01)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # scheduler = MultiStepLR(optimizer, milestones=[3, 10], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=1)
        return [optimizer], [scheduler]
        # return optimizer

    # from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/learning_rate/poly_lr.py#L16
    def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
        return initial_lr * (1 - epoch / max_epochs) ** exponent

    def prepare_batch(self, batch):
        inputs, targets = batch["img"][DATA], batch["label"][DATA]
        if torch.isnan(inputs).any():
            print("there is nan in input data!")
            inputs[inputs != inputs] = 0
        if torch.isnan(targets).any():
            print("there is nan in targets data!")
            targets[targets != targets] = 0
        # making the label as binary, it is very strange because if the label is not binary
        # the whole model cannot learn at all
        target_bin = torch.zeros(size=targets.size()).type_as(inputs)
        target_bin[targets > 0.5] = 1
        return inputs, target_bin

    def training_step(self, batch, batch_idx):
        inputs, targets = self.prepare_batch(batch)
        # print(f"training input range: {torch.min(inputs)} - {torch.max(inputs)}")
        logits = self(inputs)
        probs = torch.sigmoid(logits)
        dice, iou, _, _ = get_score(probs, targets)
        loss = dice_loss(probs, targets)
        # if self.max_loss_img["loss"] < loss.item():
        #     self.max_loss_img["loss"] = loss.item()
        #     self.max_loss_img["img"] = inputs
        #     self.max_loss_img["label"] = targets
        #     self.max_loss_img["prob"] = probs
        #     self.max_loss_img["batch_idx"] = batch_idx
        #     self.max_loss_img["filename"] = batch['filename'][0]
        # if self.min_loss_img["loss"] > loss.item():
        #     self.min_loss_img["loss"] = loss.item()
        #     self.min_loss_img["img"] = inputs
        #     self.min_loss_img["label"] = targets
        #     self.min_loss_img["prob"] = probs
        #     self.min_loss_img["batch_idx"] = batch_idx
        #     self.min_loss_img["filename"] = batch['filename'][0]
        # rand = random.randint(1, 180)
        # if batch_idx != 0 and batch_idx == rand:  # save total about 10 picture
        #     input = inputs.chunk(inputs.size()[0], 0)[0]  # split into 1 in the dimension 0
        #     target = targets.chunk(targets.size()[0], 0)[0]  # split into 1 in the dimension 0
        #     logit = probs.chunk(logits.size()[0], 0)[0]  # split into 1 in the dimension 0
        #     log_all_info(self, input, target, logit, batch_idx, "training")
        # tensorboard_logs = {"train_loss": loss, "train_IoU": iou, "train_dice": dice}
        # return {'loss': loss, "log": tensorboard_logs}
        return {'loss': loss}

    # def training_epoch_end(self, outputs):
        # log_all_info(self, self.max_loss_img["img"], self.max_loss_img["label"], self.max_loss_img["prob"],
        #              self.max_loss_img["batch_idx"], f"training: max_loss filename:{self.max_loss_img['filename']}")
        # log_all_info(self, self.min_loss_img["img"], self.min_loss_img["label"], self.min_loss_img["prob"],
        #              self.min_loss_img["batch_idx"], f"training: min_loss filename:{self.min_loss_img['filename']}")
        # loss = F.binary_cross_entropy_with_logits(logits, targets)
        # avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        # self.max_loss_img = {
        #     "loss": 0,
        #     "img": None,
        #     "label": None,
        #     "prob": None,
        #     "batch_idx": None,
        # }
        # self.min_loss_img = {
        #     "loss": 1,
        #     "img": None,
        #     "label": None,
        #     "prob": None,
        #     "batch_idx": None,
        # }
        # return {"avg_loss": avg_loss}

    def validation_step(self, batch, batch_id):
        inputs, targets = self.prepare_batch(batch)
        # print(f"input shape: {inputs.shape}, targets shape: {targets.shape}")
        # print(f"validation input range: {torch.min(inputs)} - {torch.max(inputs)}")
        logits = self(inputs)
        probs = torch.sigmoid(logits)  # compare the position
        # print(f"probs shape: {probs.shape}")
        # loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss = dice_loss(probs, targets)
        dice, iou, sensitivity, specificity = get_score(probs, targets)
        return {'val_step_loss': loss,
                'val_step_dice': dice,
                'val_step_IoU': iou,
                "val_step_sensitivity": sensitivity,
                "val_step_specificity": specificity
                }

    # Called at the end of the validation epoch with the outputs of all validation steps.
    def validation_epoch_end(self, outputs):
        # torch.stack: Concatenates sequence of tensors along a new dimension.
        avg_loss = torch.stack([x['val_step_loss'] for x in outputs]).mean()
        avg_val_dice = torch.stack([x['val_step_dice'] for x in outputs]).mean()
        # tensorboard_logs = {
        #     "val_loss": outputs[0]['val_step_loss'],  # the outputs is a dict wrapped in a list
        #     "val_dice": outputs[0]['val_step_dice'],
        #     "val_IoU": outputs[0]['val_step_IoU'],
        #     "val_sensitivity": outputs[0]['val_step_sensitivity'],
        #     "val_specificity": outputs[0]['val_step_specificity']
        # }
        # return {"loss": avg_loss, "val_dice": avg_val_dice, 'log': tensorboard_logs}
        return {"loss": avg_loss, "val_dice": avg_val_dice}

    def test_step(self, batch, batch_idx):
        inputs, targets = self.prepare_batch(batch)
        # print(f"training input range: {torch.min(inputs)} - {torch.max(inputs)}")
        logits = self(inputs)
        logits = F.interpolate(logits, size=logits.size()[2:])
        probs = torch.sigmoid(logits)
        dice, iou, _, _ = get_score(probs, targets)
        loss = dice_loss(probs, targets)
        # if batch_idx != 0 and batch_idx % 50 == 0:  # save total about 10 picture
        #     input = inputs.chunk(inputs.size()[0], 0)[0]  # split into 1 in the dimension 0
        #     target = targets.chunk(targets.size()[0], 0)[0]  # split into 1 in the dimension 0
        #     logit = probs.chunk(logits.size()[0], 0)[0]  # split into 1 in the dimension 0
        #     log_all_info(self, input, target, logit, batch_idx, "testing")
        # loss = F.binary_cross_entropy_with_logits(logits, targets)
        dice, iou, sensitivity, specificity = get_score(probs, targets)
        return {'test_step_loss': loss,
                'test_step_dice': dice,
                'test_step_IoU': iou,
                'test_step_sensitivity': sensitivity,
                'test_step_specificity': specificity
                }

    def test_epoch_end(self, outputs):
        # torch.stack: Concatenates sequence of tensors along a new dimension.
        avg_loss = torch.stack([x['test_step_loss'] for x in outputs]).mean()
        avg_dice = torch.stack([x['test_step_dice'] for x in outputs]).mean()
        avg_IoU = torch.stack([x['test_step_IoU'] for x in outputs]).mean()
        avg_sensitivity = torch.stack([x['test_step_sensitivity'] for x in outputs]).mean()
        avg_specificity = torch.stack([x['test_step_specificity'] for x in outputs]).mean()
        print(
            "test result:",
            f"avg loss: {avg_loss}",
            f"avg dice: {avg_dice}",
            f"avg IoU: {avg_IoU}",
            f"avg sensitivity: {avg_sensitivity}",
            f"avg specificity: {avg_specificity}", sep='\n'
        )
        # tensorboard_logs = {
        #     "test_loss": outputs[0]['test_step_loss'],  # the outputs is a dict wrapped in a list
        #     "test_dice": outputs[0]['test_step_dice'],
        #     "test_IoU": outputs[0]['test_step_IoU'],
        #     "test_sensitivity": outputs[0]['test_step_sensitivity'],
        #     "test_specificity": outputs[0]['test_step_specificity']
        # }
        # return {'log': tensorboard_logs}
        return {}

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        parameters defined here will be available to the model through self.hparams
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=2, help='Batch size', dest='batch_size')
        # parser.add_argument("--learning_rate", type=float, default=1e-3, help='Learning rate')
        # parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay in optimizer")
        # parser.add_argument("--down_sample", type=str, default="max", help="the way to down sample")
        # parser.add_argument("--loss", type=str, default="BCEWL", help='Loss Function')
        return parser
