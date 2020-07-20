from typing import Optional

import torch.nn as nn


class ConvolutionalBlock(nn.Module):
    def __init__(
            self,
            dimensions: int,
            in_channels: int,
            out_channels: int,
            normalization: Optional[str] = None,
            kernal_size: int = 5,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            dropout: float = 0.3,
            ):
        super().__init__()

        block = nn.ModuleList()

        # dilation = 1 if dilation is None else dilation
        # if padding:
        #     total_padding = kernel_size + 2 * (dilation - 1) - 1
        #     padding = total_padding // 2

        class_name = 'Conv{}d'.format(dimensions)
        conv_class = getattr(nn, class_name)
        conv_layer = None
        conv_layer1 = None
        norm_layer1 = None
        activation_layer1 = None
        if kernal_size == 5:
            conv_layer = conv_class(
                in_channels,
                out_channels,
                kernal_size,
                padding=(kernal_size + 1) // 2 - 1,
                padding_mode=padding_mode,
            )
        elif kernal_size == 3:
            conv_layer1 = conv_class(
                in_channels,
                out_channels,
                kernal_size,
                padding=(kernal_size + 1) // 2 - 1,
                padding_mode=padding_mode,
            )
            conv_layer = conv_class(
                out_channels,
                out_channels,
                kernal_size,
                padding=(kernal_size + 1) // 2 - 1,
                padding_mode=padding_mode,
            )
            if normalization is not None:
                if normalization == 'Batch':
                    class_name = '{}Norm{}d'.format(
                        normalization.capitalize(), dimensions)
                    norm_class = getattr(nn, class_name)
                    # num_features = in_channels if preactivation else out_channels
                    norm_layer1 = norm_class(out_channels)
                elif normalization == 'Group':
                    class_name = '{}Norm'.format(
                        normalization.capitalize())
                    norm_class = getattr(nn, class_name)
                    # num_features = in_channels if preactivation else out_channels
                    norm_layer1 = norm_class(num_groups=1, num_channels=out_channels)
                elif normalization == "InstanceNorm3d":
                    class_name = normalization
                    norm_class = getattr(nn, class_name)
                    # num_features = in_channels if preactivation else out_channels
                    norm_layer1 = norm_class(num_features=out_channels, affine=True, track_running_stats=True)

            if activation is not None:
                activation_layer1 = getattr(nn, activation)()
        elif kernal_size == 1:
            conv_layer = conv_class(
                in_channels,
                out_channels,
                kernal_size,
                padding=(kernal_size + 1) // 2 - 1,
                padding_mode=padding_mode,
            )

        norm_layer = None
        if normalization is not None:
            if normalization == 'Batch':
                class_name = '{}Norm{}d'.format(
                    normalization.capitalize(), dimensions)
                norm_class = getattr(nn, class_name)
                # num_features = in_channels if preactivation else out_channels
                norm_layer = norm_class(out_channels)
            elif normalization == 'Group':
                class_name = '{}Norm'.format(
                    normalization.capitalize())
                norm_class = getattr(nn, class_name)
                # num_features = in_channels if preactivation else out_channels
                norm_layer = norm_class(num_groups=1, num_channels=out_channels)
            elif normalization == "InstanceNorm3d":
                class_name = normalization
                norm_class = getattr(nn, class_name)
                # num_features = in_channels if preactivation else out_channels
                norm_layer = norm_class(num_features=out_channels, affine=True, track_running_stats=True)

        activation_layer = None
        if activation is not None:
            activation_layer = getattr(nn, activation)()

        # if preactivation:
        #     self.add_if_not_none(block, norm_layer)
        #     self.add_if_not_none(block, activation_layer)
        #     self.add_if_not_none(block, conv_layer)
        # else:
        self.add_if_not_none(block, conv_layer1)
        self.add_if_not_none(block, norm_layer1)
        self.add_if_not_none(block, activation_layer1)
        self.add_if_not_none(block, conv_layer)
        self.add_if_not_none(block, norm_layer)
        self.add_if_not_none(block, activation_layer)

        dropout_layer = None
        if dropout:
            class_name = 'Dropout{}d'.format(dimensions)
            dropout_class = getattr(nn, class_name)
            dropout_layer = dropout_class(p=dropout)
            self.add_if_not_none(block, dropout_layer)

        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        # self.dropout_layer = dropout_layer

        # A Sequential object runs each of the modules contained within it, in a sequential manner. This is a simpler
        # way of writing our neural network.
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

    @staticmethod
    def add_if_not_none(module_list, module):
        if module is not None:
            module_list.append(module)
