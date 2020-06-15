import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from LitUnet import LitUnet
from data.transforms import get_val_transform
from data.config import Option
import multiprocessing
import torchio as tio
from torchio import DATA


def get_output_file(input_files):
    """
    if not input the output filename, it could build it

    :param input_files:
    :return:
    """
    out_files = []
    output_folder = Path(args.folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if not args.output:
        for img in args.input:
            cur_path = output_folder / img
            out_files.append(cur_path)

    assert len(out_files) == len(in_files)
    return out_files


def get_filename(input_paths):
    for path in input_paths:
        yield os.path.split(path)


if __name__ == "__main__":
    option = Option()
    args = option.parse()
    in_files = args.input
    out_files = get_output_file(args.input)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    logging.info("Loading model ...")
    # model = LitUnet.load_from_checkpoint('./log/checkpoint/1.pth')
    model = LitUnet().cuda()
    model.eval()
    logging.info("Model loaded !")

    subjects = [
        tio.Subject(
            img=tio.Image(path=file, label=tio.INTENSITY),
        ) for file in in_files
    ]

    val_transform = get_val_transform()
    val_imageDataset = tio.ImagesDataset(subjects, transform=val_transform)
    validation_loader = torch.utils.data.DataLoader(
        val_imageDataset,
        batch_size=1,
        num_workers=multiprocessing.cpu_count(),
    )

    for index, batch in enumerate(validation_loader):
        inputs = batch["img"][DATA].to(device)
        print(inputs.shape)
        # logits = model(inputs)
        # probabilities = torch.sigmoid(logits)
        #
        # mask = probabilities > 0.5
        # print(f"mask shape: {mask.shape}")
