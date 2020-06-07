import torch
from torchio import AFFINE, DATA, PATH, TYPE, STEM
from data.get_datasets import get_dataset
import warnings
import torchio
import numpy as np
from utils.unet import UNet, UNet3D
from data.const import *
from utils.loss import get_dice_loss
from data.config import Option
import enum
import SimpleITK as sitk
import multiprocessing
import nibabel as nib
from time import ctime
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
# from torch.nn import CrossEntropyLoss  # dont work
from torchvision.transforms import Resize
from utils.matrixes import matrix
import argparse
import logging
import sys
import os


# those words used when building the dataset subject
img = "img"
label = "label"
dir_checkpoint = 'checkpoint/'


class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'


def prepare_batch(batch, device):
    inputs = batch[img][DATA].to(device)
    foreground = batch[label][DATA].to(device).squeeze()
    targets = torch.zeros_like(foreground).to(device)
    targets[foreground > 0.5] = 1
    return inputs, targets


def forward(model, inputs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        logits = model(inputs)
    return logits


def get_model_and_optimizer(device):
    model = UNet(
        in_channels=1,
        out_classes=1,
        dimensions=3,
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        # normalization='batch',
        upsampling_type='conv',
        padding=2,
        activation='PReLU',
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    return model, optimizer


def run_epoch(epoch_idx, action, loader, model, optimizer, min_loss):
    is_training = action == Action.TRAIN
    epoch_losses = []
    model.train(is_training)
    ious = []
    dices = []
    i = 0
    for batch_idx, batch in enumerate(loader):
        i += 1
        inputs, targets = prepare_batch(batch, device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            logits = forward(model, inputs)
            probabilities = torch.sigmoid(logits)
            iou, dice, batch_loss = matrix(probabilities, targets)
            ious.append(iou)
            dices.append(dice)
            # batch_loss = batch_losses.mean()
            if is_training:
                batch_loss.backward()
                optimizer.step()
            epoch_losses.append(batch_loss)
    epoch_losses = np.array(epoch_losses)
    ious = np.array(ious)
    dices = np.array(dices)
    print(f'{ctime()}: Epoch: {epoch_idx} | {action.value} mean loss: {epoch_losses.mean():0.5f} | iou: {ious.mean():0.5f} | dices : {dices.mean():0.5f}')
    if action.value == Action.VALIDATE and epoch_losses.mean() < min_loss:
        min_loss = epoch_losses
        torch.save(model.state_dict(), f'Epoch_{epoch_idx}_loss_{min_loss}.pth')
        logging.info(f'{ctime()} :Saved interrupt')


def train(num_epochs, training_loader, validation_loader, model, optimizer, min_loss):
    run_epoch(0, Action.VALIDATE, validation_loader, model, optimizer, min_loss)
    for epoch_idx in range(1, num_epochs + 1):
        print('Starting epoch', epoch_idx)
        run_epoch(epoch_idx, Action.TRAIN, training_loader, model, optimizer, min_loss)
        run_epoch(epoch_idx, Action.VALIDATE, validation_loader, model, optimizer, min_loss)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    option = Option()
    args = option.parse()

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device {device}')
    CHANNELS_DIMENSION = 1
    SPATIAL_DIMENSIONS = 2, 3, 4

    training_batch_size = args.batchsize
    validation_batch_size = args.batchsize
    num_epochs = 500

    # datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR, ADNI_DATASET_DIR_1]
    datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR]
    training_set, validation_set = get_dataset(datasets)

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=training_batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        # num_workers=0,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=validation_batch_size,
        num_workers=multiprocessing.cpu_count(),
        # num_workers=0,
    )

    model, optimizer = get_model_and_optimizer(device)
    logging.info(f'get Network!\n')

    if args.load:
        model.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    model.to(device=device)
    min_loss = 1000

    try:
        train(
            num_epochs=args.epochs,
            training_loader=training_loader,
            validation_loader=validation_loader,
            model=model,
            optimizer=optimizer,
            min_loss=min_loss)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), './checkpoint/INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
