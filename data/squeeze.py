import torch
import numpy as np
import nibabel as nib
from torchio.data.subject import Subject
from torchio import DATA, AFFINE
from torchio.transforms import Transform


class ToSqueeze(Transform):
    """Squeese the ADNI label image, if not it will have this error:

    sitk::ERROR: Pixel type: vector of 32-bit float is not supported in 3D byclass itk::simple::ResampleImageFilter
    """

    def apply_transform(self, sample: Subject) -> dict:
        for image_dict in sample.get_images(intensity_only=False):
            image_dict[DATA] = image_dict[DATA].squeeze().unsqueeze(0)
        return sample
