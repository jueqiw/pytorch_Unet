import torch
from torchio import AFFINE, DATA, PATH, TYPE, STEM
from data.get_dataset import get_dataset
import warnings
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Pad,
    Compose,
)

import torchio
import numpy as np
from utils.unet import UNet, UNet3D
from data.const import *
import enum
import SimpleITK as sitk
import multiprocessing
import nibabel as nib
from time import ctime
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
from torchvision.transforms import Resize
from torchsummary import summary

# those words used when building the dataset subject
img = "img"
label = "label"

class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'


def prepare_batch(batch, device):
    inputs = batch[img][DATA].to(device)
    foreground = batch[label][DATA].to(device)
    foreground[foreground > 0.5] = 1
    background = 1 - foreground
    targets = torch.cat((background, foreground), dim=CHANNELS_DIMENSION)
    return inputs, targets


def get_dice_score(output, target, epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score


def get_dice_loss(output, target):
    return 1 - get_dice_score(output, target)


def forward(model, inputs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        logits = model(inputs)
    return logits


def get_model_and_optimizer(device):
    model = UNet(
        in_channels=1,
        out_classes=2,
        dimensions=3,
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        # normalization='batch',
        upsampling_type='linear',
        padding=True,
        activation='PReLU',
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    return model, optimizer


def run_epoch(epoch_idx, action, loader, model, optimizer):
    is_training = action == Action.TRAIN
    epoch_losses = []
    model.train(is_training)
    for batch_idx, batch in enumerate(loader):
        inputs, targets = prepare_batch(batch, device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            logits = forward(model, inputs)
            probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
            batch_losses = get_dice_loss(probabilities, targets)
            batch_loss = batch_losses.mean()
            if is_training:
                batch_loss.backward()
                optimizer.step()
            epoch_losses.append(batch_loss.item())
    epoch_losses = np.array(epoch_losses)
    print(f'{action.value} mean loss: {epoch_losses.mean():0.3f}')


def train(num_epochs, training_loader, validation_loader, model, optimizer, weights_stem):
    run_epoch(0, Action.VALIDATE, validation_loader, model, optimizer)
    for epoch_idx in range(1, num_epochs + 1):
        print('Starting epoch', epoch_idx)
        run_epoch(epoch_idx, Action.TRAIN, training_loader, model, optimizer)
        run_epoch(epoch_idx, Action.VALIDATE, validation_loader, model, optimizer)
        torch.save(model.state_dict(), f'{weights_stem}_epoch_{epoch_idx}.pth')


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    CHANNELS_DIMENSION = 1
    SPATIAL_DIMENSIONS = 2, 3, 4

    # datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR, ADNI_DATASET_DIR_1]
    datasets = [CC359_DATASET_DIR]
    subjects = get_dataset(datasets)

    training_transform = Compose([
        RescaleIntensity((0, 1)),  # so that there are no negative values for RandomMotion
        RandomMotion(),
        # HistogramStandardization(landmarks_dict={MRI: landmarks}),
        RandomBiasField(),
        ZNormalization(masking_method=ZNormalization.mean),
        RandomNoise(),
        ToCanonical(),
        CropOrPad((240, 240, 240)),  # do not know what it do
        RandomFlip(axes=(0,)),
        OneOf({
            RandomAffine(): 0.8,
            RandomElasticDeformation(): 0.2,
        }),
    ])

    validation_transform = Compose([
        # HistogramStandardization(landmarks_dict={MRI: landmarks}),
        ZNormalization(masking_method=ZNormalization.mean),
        ToCanonical(),
        CropOrPad((240, 240, 240)),
        # Resample((4, 4, 4)),
    ])

    num_subjects = len(subjects)
    print(f"{ctime}: get total number of {num_subjects} subjects")
    num_training_subjects = int(num_subjects * 0.9)  # （5074+359+21） * 0.9 used for training

    training_subjects = subjects[:num_training_subjects]
    validation_subjects = subjects[num_training_subjects:]

    training_set = torchio.ImagesDataset(
        training_subjects, transform=training_transform)

    validation_set = torchio.ImagesDataset(
        validation_subjects, transform=validation_transform)

    print('Training set:', len(training_set), 'subjects')
    print('Validation set:', len(validation_set), 'subjects')

    training_batch_size = 2
    validation_batch_size = 1
    num_epochs = 5

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

    # one_batch = next(iter(training_loader))
    #
    # k = 4
    # batch_mri = one_batch[img][DATA][..., k]
    # batch_label = one_batch[label][DATA][..., k]
    # slices = torch.cat((batch_mri, batch_label))
    # image_path = 'batch_whole_images.png'
    # save_image(slices, image_path, nrow=training_batch_size // 2, normalize=True, scale_each=True)

    model, optimizer = get_model_and_optimizer(device)

    # summary(model, (2, 4, 37, 37))

    weights_stem = 'whole_images'
    train(num_epochs, training_loader, validation_loader, model, optimizer, weights_stem)

    batch = next(iter(validation_loader))
    model.eval()
    inputs, targets = prepare_batch(batch, device)
    with torch.no_grad():
        logits = forward(model, inputs)

