"""code is from: https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
"""
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from glob import glob
from data.const import CROPPED_LABEL, CROPPED_IMG
from typing import Tuple, List, Text
from matplotlib.pyplot import Axes, Figure
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
from matplotlib.text import Text
import imageio
from matplotlib import animation
import os
import re
# need to import this to show img in pycharm
import pylab


if __name__ == "__main__":
    img_path_list = sorted([
        Path(f) for f in sorted(glob(f"{str(CROPPED_IMG)}/**/*.nii*", recursive=True))
    ])
    label_path_list = sorted([
        Path(f) for f in sorted(glob(f"{str(CROPPED_LABEL)}/**/*.nii.gz", recursive=True))
    ])

    for img_path, label_path in zip(img_path_list, label_path_list):
        # get the file name
        _, filename = os.path.split(img_path)
        name = re.search('(.*?).nii|(.*?).nii.gz', filename).group(1)

        img, label = nib.load(img_path), nib.load(label_path)

        data_np = img.get_data().astype(np.uint8)
        label_np = label.get_data().squeeze().astype(np.float)

        data_np = np.where(label_np > 0.5, 255, data_np)
        save_folder = Path(__file__).resolve().parent / f'gif/{name}'

        if not os.path.exists(save_folder):
            print(save_folder)
            os.mkdir(save_folder)

        lens = data_np.shape[0], data_np.shape[1], data_np.shape[2]
        frames = [[], [], []]

        frames[0] = data_np[::10]

        idx = 0
        while idx < lens[1]:
            frames[1].append(data_np[:, idx, :])
            idx += 10

        idx = 0
        while idx < lens[2]:
            frames[2].append(data_np[:, :, idx])
            idx += 10

        print('Begin saving gif')
        for i in range(0, 3):
            imageio.mimsave(save_folder / f"{i+1}.gif", frames[i], 'GIF-FI', duration=0.1)
        print("Finish!")