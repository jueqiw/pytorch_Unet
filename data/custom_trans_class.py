import torch
import numpy as np
import nibabel as nib
from torchio.data.subject import Subject
from torchio import DATA, AFFINE
from torchio.transforms import Transform
import torch.nn.functional as F
from .const import SIZE


class ToSqueeze(Transform):
    """Squeese the ADNI label image, if not it will have this error:
    sitk::ERROR: Pixel type: vector of 32-bit float is not supported in 3D byclass itk::simple::ResampleImageFilter
    """
    def apply_transform(self, sample: Subject) -> dict:
        for image_dict in sample.get_images(intensity_only=False):
            image_dict[DATA] = image_dict[DATA].squeeze().unsqueeze(0)
        return sample


class ToResize(Transform):
    """Resize the image
    """
    def apply_transform(self, sample: Subject) -> dict:
        for image_dict in sample.get_images(intensity_only=False):
            image_dict[DATA] = F.interpolate(image_dict[DATA].unsqueeze(0), size=(SIZE, SIZE, SIZE))
            image_dict[DATA] = image_dict[DATA].squeeze(0)
        return sample


class ToResize_only_image(Transform):
    """Only Resize the MR image
    """
    def apply_transform(self, sample: Subject) -> dict:
        for image_dict in sample.get_images(intensity_only=False):
            image_dict[DATA] = F.interpolate(image_dict[DATA].unsqueeze(0), size=(SIZE, SIZE, SIZE))
            image_dict[DATA] = image_dict[DATA].squeeze(0)
        return sample
