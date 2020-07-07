"""try to use the whole dataset to predict, and test the result
"""
from data.const import DATA_ROOT
from data.transform import get_train_transforms
from torchio import DATA
from pathlib import Path
from glob import glob
import torchio as tio
from torch.utils.data import DataLoader
import torch
import os


def _prepare_data(batch):
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

if __name__ == "__main__":
    img_path_folder = DATA_ROOT / "all_different_size_img" / "cropped" / "img"
    label_path_folder = DATA_ROOT / "all_different_size_img" / "cropped" / "label"

    img_path_list = sorted([
        Path(f) for f in sorted(glob(f"{str(img_path_folder)}/**/*.nii.gz", recursive=True))
    ])
    label_path_list = sorted([
        Path(f) for f in sorted(glob(f"{str(label_path_folder)}/**/*.nii.gz", recursive=True))
    ])

    subjects = []
    for img_path, label_path in zip(img_path_list, label_path_list):
        subject = tio.Subject(
                img=tio.Image(path=img_path, type=tio.INTENSITY),
                label=tio.Image(path=label_path, type=tio.LABEL),
            )
        subjects.append(subject)

    print(f"get {len(subjects)} of subject!")

    training_transform = get_train_transforms()

    training_set = tio.ImagesDataset(
        subjects, transform=training_transform)

    loader = DataLoader(training_set,
                        batch_size=2,
                        # num_workers=multiprocessing.cpu_count())
                        num_workers=8)

    for batch_idx, batch in enumerate(loader):
        inputs, targets = _prepare_data(batch)
        print(f"inputs shape: {inputs.shape}")
        print(f"targets shape: {targets.shape}")
