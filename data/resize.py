import numpy as np
from skimage.transform import resize
from .const import *


class Resize(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        items = sample.get_images_dict(intensity_only=False).items()
        for image_name, image_dict in items:
            data = image_dict['data']
            print(data.shape)
            data = resize(data, output_shape=(SIZE, SIZE, SIZE), mode='constant', anti_aliasing=True)
            image_dict['data'] = data
        return sample
