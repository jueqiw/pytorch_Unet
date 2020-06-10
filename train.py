import torch
from torchio import DATA
from data.get_datasets import get_dataset
from utils.unet import UNet, UNet3D
from data.const import *
from data.config import Option
from pathlib import Path
from postprocess.visualize import BrainSlices
import enum
import SimpleITK as sitk
import multiprocessing
import nibabel as nib
from time import ctime
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.nn import MultiLabelSoftMarginLoss
from torch.utils.tensorboard import SummaryWriter
from utils.loss import get_dice_score
import warnings
warnings.filterwarnings("ignore")
import torchio
import numpy as np
import logging
import sys
import os

# those words used when building the dataset subject
img = "img"
label = "label"
dir_checkpoint = 'checkpoint/'
option = Option()
args = option.parse()

class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'


def prepare_batch(batch, device):
    inputs = batch[img][DATA].to(device)
    foreground = batch[label][DATA].to(device)
    # dataset = batch["dataset"]
    # targets = torch.zeros_like(foreground).to(device)
    # targets[foreground > 0.5] = 1
    inputs = F.interpolate(inputs, (64, 64, 64))
    foreground = F.interpolate(foreground, (64, 64, 64))
    # print(dataset)
    return inputs, foreground


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
        normalization='Group',
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        upsampling_type='conv',
        padding=2,
        activation='PReLU',
        dropout=0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    return model, optimizer


def run_epoch(epoch_idx, action, loader, model, optimizer, min_loss, writer):
    is_training = action == Action.TRAIN
    epoch_losses = []
    model.train(is_training)
    # loss_f_mean = MultiLabelSoftMarginLoss()
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
            dice, iou = get_dice_score(probabilities, targets)
            if int(batch_idx) != 0 and args.plots and int(batch_idx) % 15 == 0:
                slices = BrainSlices(inputs, targets, logits)
                slices.visualize(int(batch_idx), epoch_idx, outdir=Path(__file__).resolve().parent / "log" / "plot")
            batch_loss = F.binary_cross_entropy_with_logits(logits, targets)
            ious.append(iou.item())
            dices.append(dice.item())
            if is_training:
                batch_loss.backward()
                optimizer.step()
            epoch_losses.append(batch_loss.item())
            print(f'{ctime()}: Epoch: {epoch_idx} Batch: {batch_idx}| {action.value} loss: {batch_loss.item():0.5f} | iou: {iou.item():0.5f} | dices : {dice.item():0.5f}')
            # if action.value == Action.TRAIN and batch_loss.item() < min_loss:
            if is_training and batch_loss.item() < min_loss:
                if os.path.exists(f"./log/checkpoint/Epoch_{epoch_idx}_loss_{min_loss:0.3}.pth"):
                    os.system(f"del ./log/checkpoint/Epoch_{epoch_idx}_loss_{min_loss:0.3}.pth")
                min_loss = batch_loss.item()
                torch.save(model.state_dict(), f'./log/checkpoint/Epoch_{min_loss}_loss_{min_loss:0.3}.pth')
                logging.info(f'{ctime()} :Saved model!')
    epoch_losses = np.array(epoch_losses)
    ious = np.array(ious)
    dices = np.array(dices)
    print(f'{ctime()}: Epoch: {epoch_idx} | {action.value} mean loss: {epoch_losses.mean():0.5f} | iou: {ious.mean():0.5f} | dices : {dices.mean():0.5f}')
    if not is_training and epoch_losses.mean() < min_loss:
        min_loss = epoch_losses.item()
        print("Yes")
        torch.save(model.state_dict(), f'./checkpoint/Epoch_{epoch_idx}_loss_{min_loss:0.3f}.pth')
        logging.info(f'{ctime()} :Saved model')
    return epoch_losses.mean(), min_loss


def train(num_epochs, training_loader, validation_loader, model, optimizer, min_loss, writer):
    for epoch_idx in range(1, num_epochs + 1):
        print('Starting epoch', epoch_idx)
        loss, min_loss = run_epoch(epoch_idx, Action.TRAIN, training_loader, model, optimizer, min_loss, writer)
        # ...log the running loss
        writer.add_scalar('training loss', loss, num_epochs * len(training_loader) + epoch_idx)
        loss, min_loss = run_epoch(epoch_idx, Action.VALIDATE, validation_loader, model, optimizer, min_loss, writer)
        writer.add_scalar('val loss', loss, num_epochs * len(validation_loader) + epoch_idx)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if not os.path.exists('./log'):
        os.mkdir('log')
    if not os.path.exists('./log/summary'):
        os.mkdir('./log/summary')
    if not os.path.exists('./log/checkpoint'):
        os.mkdir('./logcheckpoint')

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('./log/summary/Unet')

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device {device}')
    CHANNELS_DIMENSION = 1
    SPATIAL_DIMENSIONS = 2, 3, 4

    if COMPUTECANADA:
        # datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR, ADNI_DATASET_DIR_1]
        datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR]
    else:
    # datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR]
        datasets = [CC359_DATASET_DIR]

    training_set, validation_set = get_dataset(datasets)

    # Pytorch's DataLoader is responsible for managing batches. You can create a DataLoader from any Dataset.
    # DataLoader makes it easier to iterate over batches. Rather than having to use train_ds[i*bs : i*bs+bs],
    # the DataLoader gives us each minibatch automatically.
    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=2 * args.batchsize,
        num_workers=multiprocessing.cpu_count(),
    )

    model, optimizer = get_model_and_optimizer(device)
    logging.info(f'get Network!\n')


    if args.load:
        model.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    model.to(device=device)
    min_loss = 100000.0

    try:
        train(
            num_epochs=args.epochs,
            training_loader=training_loader,
            validation_loader=validation_loader,
            model=model,
            optimizer=optimizer,
            min_loss=min_loss,
            writer=writer,
            )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), './checkpoint/INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
