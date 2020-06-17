from typing import Optional

import torch
import torch.nn as nn
from torch.nn import ConvTranspose3d
import torch.nn.functional as F
from torch import Tensor

from .conv import ConvolutionalBlock

CHANNELS_DIMENSION = 1
UPSAMPLING_MODES = (
    'nearest',
    'linear',
    'bilinear',
    'bicubic',
    'trilinear',
)


class Decoder(nn.Module):
    def __init__(
            self,
            in_channels_skip_connection: int,  # 32
            num_decoding_blocks: int,
            normalization: Optional[str],
            upsampling_type: str = "conv",
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            # initial_dilation: Optional[int] = None,
            dropout: float = 0.3,
            ):
        super().__init__()
        self.decoding_blocks = nn.ModuleList()
        # self.dilation = initial_dilation
        first_decoding_block = True
        for _ in range(num_decoding_blocks):  # 3
            decoding_block = DecodingBlock(
                in_channels_skip_connection,
                upsampling_type,
                padding=padding,
                padding_mode=padding_mode,
                # dilation=self.dilation,
                dropout=dropout,
                first_decoder_block=first_decoding_block
            )
            self.decoding_blocks.append(decoding_block)
            in_channels_skip_connection //= 2
            first_decoding_block = False
            # if self.dilation is not None:
            #     self.dilation //= 2

    def forward(self, skip_connections, x):
        zipped = zip(reversed(skip_connections), self.decoding_blocks)
        for skip_connection, decoding_block in zipped:
            x = decoding_block(skip_connection, x)
        return x


class DecodingBlock(nn.Module):
    def __init__(
            self,
            in_channels_skip_connection: int,  # 32
            upsampling_type: str,
            normalization: Optional[str] = 'Group',
            # residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            # dilation: Optional[int] = None,
            dropout: float = 0,
            first_decoder_block: bool = True,
            ):
        super().__init__()

        # self.residual = residual

        if upsampling_type == 'conv':
            if first_decoder_block:
                in_channels = out_channels = in_channels_skip_connection
            else:
                in_channels = in_channels_skip_connection * 2
                out_channels = in_channels_skip_connection
            self.upsample = get_conv_transpose_layer(
                in_channels, out_channels)
        else:
            self.upsample = get_upsampling_layer(upsampling_type)

        in_channels_first = in_channels_skip_connection * 2
        out_channels = in_channels_skip_connection

        self.conv1 = ConvolutionalBlock(
            in_channels=in_channels_first,
            out_channels=out_channels,
            normalization=normalization,
            padding=padding,
            padding_mode=padding_mode,
            # dilation=dilation,
            dropout=dropout,
        )

        # in_channels_second = out_channels
        # self.conv2 = ConvolutionalBlock(
        #     dimensions,
        #     in_channels_second,
        #     out_channels,
        #     # normalization=normalization,
        #     # preactivation=preactivation,
        #     padding=padding,
        #     padding_mode=padding_mode,
        #     dilation=dilation,
        #     dropout=dropout,
        # )

        # if residual:
        #     self.conv_residual = ConvolutionalBlock(
        #         dimensions,
        #         in_channels_first,
        #         out_channels,
        #         kernel_size=1,
        #         # normalization=None,
        #         activation=None,
        #     )

    def forward(self, skip_connection, x):
        x = self.upsample(x)  # upConvLayer
        cropped = self.crop(x, skip_connection)
        x = torch.cat((cropped, x), dim=CHANNELS_DIMENSION)
        x = self.conv1(x)
        print(f"x.shape {x.shape}")
        return x

    def crop(self, x: Tensor, skip: Tensor) -> Tensor:
        # x.shape == skip.shape copy code from
        # https://github.com/DM-Berger/unet-learn/blob/1a197860ae8b87cb42802454ad830b9fd3c6f2e4/src/model/decode.py#L40
        # (left, right, top, bottom, front, back) is F.pad order we are just implementing our own version of
        # https://github.com/fepegar/unet/blob/9f64483d351b4f7d95c0d871aa7aa587b8fdb21b/unet/decoding.py#L142 but
        # fixing their bug which won't work for odd numbers
        shape_diffs = torch.tensor(x.shape)[2:] - torch.tensor(skip.shape)[2:]
        halfs = torch.true_divide(shape_diffs, 2)
        halfs_left = -torch.floor(halfs).to(dtype=int)
        halfs_right = -torch.ceil(halfs).to(dtype=int)
        pads = torch.stack([halfs_left, halfs_right]).t().flatten().tolist()
        cropped = F.pad(skip, pads)
        return cropped


def get_upsampling_layer(upsampling_type: str) -> nn.Upsample:
    if upsampling_type not in UPSAMPLING_MODES:
        message = (
            'Upsampling type is "{}"'
            ' but should be one of the following: {}'
        )
        message = message.format(upsampling_type, UPSAMPLING_MODES)
        raise ValueError(message)
    return nn.Upsample(scale_factor=2, mode=upsampling_type)


def get_conv_transpose_layer(in_channels, out_channels):
    conv_layer = ConvTranspose3d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
    return conv_layer
