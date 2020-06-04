from pathlib import Path
import os

SIZE = 128
LEARN_RATE = 0.001

#
# DATASET_DIR = "/project/6005889/U-Net_MRI-Data"
#
# CC359_DATASET_DIR = "CalgaryCampinas359/Original"
# CC359_LABEL_DIR = "CalgaryCampinas359/Skull-stripping-masks/STAPLE"
# CC359_MANUAL_LABEL_DIR = "CalgaryCampinas359/Skull-stripping-masks/Manual"
#
# NFBS_DATASET_DIR = "NFBS/NFBS_Dataset"
#
# ADNI_DATASET_DIR_1 = "/project/6005889/U-Net_MRI-Data/ADNI"
# ADNI_DATASET_DIR_2 = "/project/6005889/U-Net_MRI-Data/ADNI/ADNI"
#
# ADNI_LABEL = "brain_extraction"

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "work"

print(DATA_ROOT)

CC359_DATASET_DIR = DATA_ROOT / "CalgaryCampinas359//Original"
CC359_LABEL_DIR = DATA_ROOT / "CalgaryCampinas359//Skull-stripping-masks//STAPLE"
CC359_MANUAL_LABEL_DIR = DATA_ROOT / "CalgaryCampinas359//Skull-stripping-masks//Manual"

NFBS_DATASET_DIR = DATA_ROOT / "NFBS/NFBS_Dataset"

ADNI_DATASET_DIR_1 = "/project/6005889/U-Net_MRI-Data/ADNI"
ADNI_DATASET_DIR_2 = "/project/6005889/U-Net_MRI-Data/ADNI/ADNI"

ADNI_LABEL = "brain_extraction"

