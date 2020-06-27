from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Pad,
    Compose,
)
from .custom_trans_class import ToSqueeze, ToResize, ToResize_only_image


def get_train_transforms() -> Compose:
    training_transform = Compose([
        ToSqueeze(),
        ToCanonical(),
        # Resample(4.0),  # need to be written again
        # CropOrPad(64),
        RescaleIntensity((0, 1)),  # so that there are no negative values for RandomMotion
        RandomMotion(),
        # HistogramStandardization(landmarks_dict={MRI: landmarks}),
        RandomBiasField(),
        RandomNoise(),
        RandomFlip(axes=(0, 1, 2)),
        OneOf({
            RandomAffine(): 0.8,
            RandomElasticDeformation(): 0.2,
        }),
        ToResize(),
        ZNormalization(masking_method=ZNormalization.mean),  # Subtract mean and divide by standard deviation.
    ])

    return training_transform


def get_val_transform() -> Compose:
    validation_transform = Compose([
        ToSqueeze(),
        ToCanonical(),
        # Resample(4.0),
        # CropOrPad(64),
        RescaleIntensity((0, 1)),
        ToResize(),
        ZNormalization(masking_method=ZNormalization.mean),
    ])
    return validation_transform


def get_test_transform() -> Compose:
    validation_transform = Compose([
        ToSqueeze(),
        ToCanonical(),
        # Resample(4.0),
        # CropOrPad(64),
        RescaleIntensity((0, 1)),
        ToResize_only_image(),
        ZNormalization(masking_method=ZNormalization.mean),
    ])
    return validation_transform
