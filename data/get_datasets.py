from .get_subjects import *
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
from .squeeze import ToSqueeze


def get_dataset(datasets):
    # datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR, ADNI_DATASET_DIR_1]
    # datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR]
    subjects = get_subjects(datasets)

    training_transform = Compose([
        ToSqueeze(),
        RescaleIntensity((0, 1)),  # so that there are no negative values for RandomMotion
        RandomMotion(),
        # HistogramStandardization(landmarks_dict={MRI: landmarks}),
        RandomBiasField(),
        RandomNoise(),
        ToCanonical(),
        # CropOrPad((128, 128, 128)),  # do not know what it do
        RandomFlip(axes=(0,)),
        OneOf({
            RandomAffine(): 0.8,
            RandomElasticDeformation(): 0.2,
        }),
    ])

    validation_transform = Compose([
        ToCanonical(),
    ])

    num_subjects = len(subjects)
    # print(f"{ctime()}: get total number of {num_subjects} subjects")
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