"""create a MRI class for every instance, to load every instance's image and label path"""
from .const import *
import os
import re
import random
import nibabel as nib


class MRI:

    def __init__(self, dataset, file_name):
        """
        :param dataset: dataset name
        :param file_name: every instance
        """
        self.dataset = dataset
        self.file_name = file_name
        self.cc359_manual = set()
        self.img_path = ""
        self.label_path = ""
        self.val_data = random.random()
        self.flag = True  # in case some file are not NIFTI file
        self.get_path()
        # if (MRI.id % 10) == 0:
        #     print("have processed %d file" % MRI.id)
        self.cc359_manual = set()
        # files in val_label_dir are Manual label, get these files
        for f in os.listdir(CC359_LABEL_DIR):
            regex = re.compile(r'^CC(.*?)_manual')
            match_object = re.match(regex, f)
            if match_object is not None:
                self.cc359_manual.add(match_object.group(0))

    def get_path(self):
        """
        get NTFTi file path with orignial MRI and label path
        :return:
        """
        if self.dataset == CC359_DATASET_DIR:
            regex = re.compile(r"CC(.*?).nii.gz")
            self.img_path = CC359_DATASET_DIR / self.file_name

            if regex.match(self.file_name) and self.file_name[0] != '.':
                name = self.file_name[:self.file_name.find('.')]

                if name not in self.cc359_manual:
                    self.label_path = CC359_LABEL_DIR / "{}_staple.nii.gz".format(name)
                else:
                    # use the manual data to get more precise
                    self.label_path = CC359_MANUAL_LABEL_DIR / "{}_manual.nii.gz".format(name)
            else:
                self.flag = False
                print("cant identify this file:", self.file_name)

        elif self.dataset == NFBS_DATASET_DIR:
            patient_path = NFBS_DATASET_DIR / self.file_name

            if os.path.isdir(patient_path) and self.file_name[0] != '.':
                self.img_path = os.path.join(patient_path, "sub-{}_ses-NFB3_T1w.nii.gz".format(self.file_name))
                self.label_path = os.path.join(patient_path,
                                               "sub-{}_ses-NFB3_T1w_brainmask.nii.gz".format(self.file_name))

        elif self.dataset == ADNI_DATASET_DIR_1 or self.dataset == ADNI_DATASET_DIR_2:
            self.img_path = self.file_name
            label_file_name = str(self.file_name.name + '.gz')
            self.label_path = os.path.join(ADNI_DATASET_DIR_1, ADNI_LABEL, label_file_name)
            if not (os.path.exists(self.img_path) and os.path.exists(self.label_path)):
                self.flag = False

    def show_image_shape(self):
        """
        print the shape of the MRI
        :return: None
        """
        img = []
        seg = []
        try:
            # in case some times found the file isnt exist like ".xxx" file
            img = nib.load(self.img_path).get_data()
        except OSError as e:
            print("not such img file:", self.img_path)
        try:
            seg = nib.load(self.label_path).get_data().squeeze()
        except OSError as e:
            print("not such label file:", self.label_path)
        return img, seg
