import torch
from utils.unet import UNet
from torchsummary import summary
from data.get_subjects import get_subjects
from data.const import *
from data.get_datasets import get_dataset
import multiprocessing

def get_model_and_optimizer(device):
    model = UNet(
        in_channels=1,
        out_classes=1,
        dimensions=3,
        normalization='Group',
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        upsampling_type='conv',
        padding=2,
        activation='PReLU',
        dropout=0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    return model, optimizer


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
model, optimizer = get_model_and_optimizer(device)
print(model)
