from .const import ADNI_DATASET_DIR_1, ADNI_DATASET_DIR_2, ADNI_LABEL
from .MRI import MRI
import pandas as pd
import os
from pathlib import Path
import re


def get_path(datasets):
    for dataset in datasets:
        if dataset == ADNI_DATASET_DIR_1:
            brain_label = Path(os.path.join(ADNI_DATASET_DIR_1, ADNI_LABEL))
            originals = list(Path(ADNI_DATASET_DIR_1).glob("**/*.nii"))
            originals.extend(list(Path(ADNI_DATASET_DIR_2).glob("**/*.nii")))

            fine_tune_set_file = Path(__file__).resolve().parent.parent.parent / "ADNI_MALPEM_baseline_1069.csv"
            file_df = pd.read_csv(fine_tune_set_file, sep=',')
            images_baseline_set = set(file_df['filename'])

            regex = re.compile(r"ADNI_(.*?).nii.gz")
            brain_label_set = set(label for label in os.listdir(brain_label) if regex.match(label))

            for original in originals:
                cur = Path(original)

                # print("stem:", cur.stem)  # without suffix
                # print("name:", cur.name)  # with suffix
                label_file = cur.name + ".gz"
                if label_file in brain_label_set and label_file in images_baseline_set:
                    mri = MRI(dataset, original)
                    if mri.flag:
                        yield mri
        else:
            for file_name in os.listdir(dataset):
                mri = MRI(dataset, file_name)
                yield mri


