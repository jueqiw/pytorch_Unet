from .const import ADNI_DATASET_DIR_1, ADNI_DATASET_DIR_2, ADNI_LABEL
from .MRI import MRI
import os
from pathlib import Path
import re


def get_path(datasets):
    for dataset in datasets:
        if dataset == ADNI_DATASET_DIR_1:
            brain_label = Path(os.path.join(ADNI_DATASET_DIR_1, ADNI_LABEL))
            originals = list(Path(ADNI_DATASET_DIR_1).glob("**/*.nii"))
            originals.extend(list(Path(ADNI_DATASET_DIR_2).glob("**/*.nii")))

            regex = re.compile(r"ADNI_(.*?).nii.gz")
            brain_label_set = set(label for label in os.listdir(brain_label) if regex.match(label))

            for original in originals:
                cur = Path(original)

                # print("stem:", cur.stem)  # without suffix
                # print("name:", cur.name)  # with suffix
                label_file = cur.name + ".gz"
                if label_file in brain_label_set:
                    mri = MRI(dataset, original)
                    if mri.flag:
                        yield mri
        else:
            for file_name in os.listdir(dataset):
                mri = MRI(dataset, file_name)
                yield mri


