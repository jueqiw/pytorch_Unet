# -*- coding: utf-8 -*-

"""Main module. right now just a copy from https://github.com/fepegar/unet/tree/master/unet right now"""

from typing import Optional
import torch.nn as nn
from .encoding import Encoder, EncodingBlock
from .decoding import Decoder
from .conv import ConvolutionalBlock


class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_classes: int = 1,  # 1
            num_encoding_blocks: int = 3,
            out_channels_first_layer: int = 8,
            normalization: Optional[str] = 'Group',
            pooling_type: str = 'max',
            upsampling_type: str = 'conv',
            preactivation: bool = False,
            residual: bool = False,
            padding: int = 2,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            initial_dilation: Optional[int] = None,
            dropout: float = 0.3,
            monte_carlo_dropout: float = 0.3,
            ):
        super().__init__()
        depth = num_encoding_blocks  # 3

        # Force padding if residual blocks
        if residual:
            padding = 1

        # Encoder Conv3D *2 *2
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels_first=out_channels_first_layer,
            pooling_type=pooling_type,
            num_encoding_blocks=depth,
            normalization=normalization,
            # preactivation=preactivation,
            # residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            # initial_dilation=initial_dilation,
            dropout=dropout,
        )

        in_channels = self.encoder.out_channels  # 32
        in_channels_skip_connection = in_channels  # 32

        num_decoding_blocks = depth
        self.decoder = Decoder(
            in_channels_skip_connection=in_channels_skip_connection,
            num_decoding_blocks=num_decoding_blocks,
            normalization=normalization,
            upsampling_type="conv",
            # preactivation=preactivation,
            residual=residual,
            padding=2,
            padding_mode=padding_mode,
            activation=activation,
            # initial_dilation=self.encoder.dilation,
            dropout=dropout,
        )

        # Monte Carlo dropout
        # self.monte_carlo_layer = None
        # if monte_carlo_dropout:
        #     dropout_class = getattr(nn, 'Dropout{}d'.format(dimensions))
        #     self.monte_carlo_layer = dropout_class(p=monte_carlo_dropout)

        # Classifier
        in_channels = out_channels_first_layer
        self.classifier = ConvolutionalBlock(
            in_channels, out_channels=out_classes,
            kernel_size=1, activation=None,
            dropout=0,
        )

    def forward(self, x):
        skip_connections, encoding = self.encoder(x)
        x = self.decoder(skip_connections, encoding)
        return self.classifier(x)
