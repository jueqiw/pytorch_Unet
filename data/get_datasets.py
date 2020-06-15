from .get_subjects import *
from .transform import get_val_transform, get_train_transforms
import torchio


def get_dataset(datasets):
    subjects = get_subjects(datasets)

    training_transform = get_train_transforms()
    validation_transform = get_val_transform()

    num_subjects = len(subjects)
    # print(f"{ctime()}: get total number of {num_subjects} subjects")
    num_training_subjects = int(num_subjects * 0.9)  # （5074+359+21） * 0.9 used for training

    training_subjects = subjects[:num_training_subjects]
    validation_subjects = subjects[num_training_subjects:]

    training_set = torchio.ImagesDataset(
        training_subjects, transform=training_transform)

    validation_set = torchio.ImagesDataset(
        validation_subjects, transform=validation_transform)
    return training_set, validation_set