import torch
from utils.unet import UNet
from torchsummary import summary
from data.get_subjects import get_subjects
from data.const import *
from data.get_datasets import get_dataset
import multiprocessing

# def get_model_and_optimizer(device):
#     model = UNet(
#         in_channels=1,
#         out_classes=2,
#         dimensions=3,
#         num_encoding_blocks=3,
#         out_channels_first_layer=8,
#         # normalization='batch',
#         upsampling_type='linear',
#         padding=True,
#         activation='PReLU',
#     ).to(device)
#     optimizer = torch.optim.AdamW(model.parameters())
#     return model, optimizer
#
#
# device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
# model, optimizer = get_model_and_optimizer(device)
# print(model)
# model(torch.tensor((1, 128, 128, 128)))
# summary(model, (1, 128, 128, 128))

datasets = [ADNI_DATASET_DIR_1]

t, v = get_dataset(datasets)

training_loader = torch.utils.data.DataLoader(
        t,
        batch_size=5,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
    )

validation_loader = torch.utils.data.DataLoader(
    v,
    batch_size=5,
    num_workers=multiprocessing.cpu_count(),
)




