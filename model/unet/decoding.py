from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

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
            dimensions: int,
            upsampling_type: str,
            num_decoding_blocks: int,
            normalization: Optional[str],
            # preactivation: bool = False,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            # initial_dilation: Optional[int] = None,
            dropout: float = 0.3,
            ):
        super().__init__()
        upsampling_type = fix_upsampling_type(upsampling_type, dimensions)
        self.decoding_blocks = nn.ModuleList()
        # self.dilation = initial_dilation
        first_decoding_block = True
        for i in range(num_decoding_blocks):  # 3
            decoding_block = DecodingBlock(
                in_channels_skip_connection,
                dimensions,
                upsampling_type,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                # dilation=self.dilation,
                dropout=dropout,
                first_decoder_block=first_decoding_block,
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
            dimensions: int,
            upsampling_type: str,
            normalization: Optional[str] = 'Group',
            # residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
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
                dimensions, in_channels, out_channels)
        else:
            self.upsample = get_upsampling_layer(upsampling_type)

        in_channels_first = in_channels_skip_connection * 2
        out_channels = in_channels_skip_connection

        self.conv1 = ConvolutionalBlock(
            dimensions,
            in_channels_first,
            out_channels,
            normalization=normalization,
            # preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
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
        #     activation=activation,
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
        # if self.all_size_input:
        #     x = self.crop(x, skip_connection)  # crop x according skip_connection
        x = torch.cat((skip_connection, x), dim=CHANNELS_DIMENSION)
        x = self.conv1(x)
        return x

    def crop(self, x: Tensor, skip: Tensor) -> Tensor:
        # Code is from
        # https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py#L57
        # but change 2D to 3D

        diffT = skip.size()[2] - x.size()[2]
        diffH = skip.size()[3] - x.size()[3]
        diffW = skip.size()[4] - x.size()[4]
        return x


def get_upsampling_layer(upsampling_type: str) -> nn.Upsample:
    if upsampling_type not in UPSAMPLING_MODES:
        message = (
            'Upsampling type is "{}"'
            ' but should be one of the following: {}'
        )
        message = message.format(upsampling_type, UPSAMPLING_MODES)
        raise ValueError(message)
    return nn.Upsample(scale_factor=2, mode=upsampling_type)


def get_conv_transpose_layer(dimensions, in_channels, out_channels):
    class_name = 'ConvTranspose{}d'.format(dimensions)
    conv_class = getattr(nn, class_name)
    conv_layer = conv_class(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
    return conv_layer


def fix_upsampling_type(upsampling_type: str, dimensions: int):  # ??
    if upsampling_type == 'linear':
        if dimensions == 2:
            upsampling_type = 'bilinear'
        elif dimensions == 3:
            upsampling_type = 'trilinear'
    return upsampling_type