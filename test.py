import torch
from utils.unet import UNet
from torchsummary import summary


def get_model_and_optimizer(device):
    model = UNet(
        in_channels=1,
        out_classes=2,
        dimensions=3,
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        # normalization='batch',
        upsampling_type='linear',
        padding=True,
        activation='PReLU',
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    return model, optimizer


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
model, optimizer = get_model_and_optimizer(device)
# print(model)
# model(torch.tensor((1, 128, 128, 128)))
# summary(model, (1, 128, 128, 128))
