import torch
from torchio import AFFINE, DATA, PATH, TYPE, STEM
from data.get_datasets import get_dataset
from tqdm import tqdm
from utils.unet import UNet, UNet3D
from data.const import *
from data.config import Option
import enum
import SimpleITK as sitk
import multiprocessing
import nibabel as nib
from time import ctime
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
# from torch.nn import BCEWithLogitsLoss
from torch.nn import MultiLabelSoftMarginLoss
from torch.utils.tensorboard import SummaryWriter
from utils.matrixes import matrix
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


class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'


def prepare_batch(batch, device):
    inputs = batch[img][DATA].to(device)
    foreground = batch[label][DATA].to(device)
    targets = torch.zeros_like(foreground).to(device)
    targets[foreground > 0.5] = 1
    inputs = F.interpolate(inputs, (128, 128, 128))
    targets = F.interpolate(targets, (128, 128, 128))
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
        upsampling_type='conv',
        padding=2,
        activation='PReLU',
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    return model, optimizer


def run_epoch(epoch_idx, action, loader, model, optimizer, min_loss, writer):
    is_training = action == Action.TRAIN
    epoch_losses = []
    model.train(is_training)
    loss_f_mean = MultiLabelSoftMarginLoss()
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
            iou, dice = matrix(probabilities, targets)
            batch_loss = loss_f_mean(logits, targets)
            ious.append(iou)
            dices.append(dice)
            if is_training:
                batch_loss.backward()
                optimizer.step()
            epoch_losses.append(batch_loss.item())
    epoch_losses = np.array(epoch_losses)
    ious = np.array(ious)
    dices = np.array(dices)
    print(f'{ctime()}: Epoch: {epoch_idx} | {action.value} mean loss: {epoch_losses.mean():0.5f} | iou: {ious.mean():0.5f} | dices : {dices.mean():0.5f}')
    if action.value == Action.VALIDATE and epoch_losses.mean() < min_loss:
        min_loss = epoch_losses
        torch.save(model.state_dict(), f'./checkpoint/Epoch_{epoch_idx}_loss_{min_loss}.pth')
        logging.info(f'{ctime()} :Saved model')
    return epoch_losses.mean()


def train(num_epochs, training_loader, validation_loader, model, optimizer, min_loss, writer):
    for epoch_idx in range(1, num_epochs + 1):
        print('Starting epoch', epoch_idx)
        loss = run_epoch(epoch_idx, Action.TRAIN, training_loader, model, optimizer, min_loss, writer)
        # ...log the running loss
        writer.add_scalar('training loss', loss, num_epochs * len(training_loader) + epoch_idx)
        loss = run_epoch(epoch_idx, Action.VALIDATE, validation_loader, model, optimizer, min_loss, writer)
        writer.add_scalar('val loss', loss, num_epochs * len(validation_loader) + epoch_idx)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    option = Option()
    args = option.parse()

    if not os.path.exists('./summary'):
        os.mkdir('summary')
    if not os.path.exists('./checkpoint'):
        os.mkdir('checkpoint')

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('summary/Unet')

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device {device}')
    CHANNELS_DIMENSION = 1
    SPATIAL_DIMENSIONS = 2, 3, 4

    datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR, ADNI_DATASET_DIR_1]
    # datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR]
    # datasets = [ADNI_DATASET_DIR_1]
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
    min_loss = 100000

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
