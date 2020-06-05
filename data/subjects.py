from .get_dataset import *
import torchio
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

def get_():
    # datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR, ADNI_DATASET_DIR_1]
    datasets = [CC359_DATASET_DIR]
    subjects = get_dataset(datasets)

    training_transform = Compose([
        RescaleIntensity((0, 1)),  # so that there are no negative values for RandomMotion
        RandomMotion(),
        # HistogramStandardization(landmarks_dict={MRI: landmarks}),
        RandomBiasField(),
        ZNormalization(masking_method=ZNormalization.mean),
        RandomNoise(),
        ToCanonical(),
        # CropOrPad((240, 240, 240)),  # do not know what it do
        RandomFlip(axes=(0,)),
        OneOf({
            RandomAffine(): 0.8,
            RandomElasticDeformation(): 0.2,
        }),
    ])

    validation_transform = Compose([
        # HistogramStandardization(landmarks_dict={MRI: landmarks}),
        ZNormalization(masking_method=ZNormalization.mean),
        ToCanonical(),
        # CropOrPad((240, 240, 240)),
        # Resample((4, 4, 4)),
    ])

    num_subjects = len(subjects)
    print(f"{ctime}: get total number of {num_subjects} subjects")
    num_training_subjects = int(num_subjects * 0.9)  # （5074+359+21） * 0.9 used for training

    training_subjects = subjects[:num_training_subjects]
    validation_subjects = subjects[num_training_subjects:]

    training_set = torchio.ImagesDataset(
        training_subjects, transform=training_transform)

    validation_set = torchio.ImagesDataset(
        validation_subjects, transform=validation_transform)

    print('Training set:', len(training_set), 'subjects')
    print('Validation set:', len(validation_set), 'subjects')
    return training_set, validation_set