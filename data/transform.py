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
from .custom_trans_class import ToSqueeze, ToResize


def get_train_transforms() -> Compose:
    training_transform = Compose([
        ToSqueeze(),
        ToCanonical(),
        ZNormalization(masking_method=ZNormalization.mean),  # Subtract mean and divide by standard deviation.
        RescaleIntensity((0, 1)),  # so that there are no negative values for RandomMotion
        RandomMotion(),
        # HistogramStandardization(landmarks_dict={MRI: landmarks}),
        RandomBiasField(),
        RandomNoise(),
        RandomFlip(axes=(0,)),
        OneOf({
            RandomAffine(): 0.8,
            RandomElasticDeformation(): 0.2,
        }),
        ToResize()
    ])

    return training_transform


def get_val_transform() -> Compose:
    validation_transform = Compose([
        ToSqueeze(),
        ZNormalization(masking_method=ZNormalization.mean),
        ToCanonical(),
    ])
    return validation_transform
