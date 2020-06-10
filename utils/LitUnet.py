import pytorch_lightning as pl
from typing import Optional
from torch.nn import MultiLabelSoftMarginLoss
from .unet import *
import torch


class LitUnet(pl.LightningModule):

    def __init__(
            self,
            out_channels_first_layer: int = 8,
            out_classes: int = 1,  # also the number of labels
            dimensions: int = 3,
            num_encoding_blocks: int = 3,
            normalization: Optional[str] = "Group",
            batch_size: int = 1,
    ):
        super().__init__()
        self.unet = UNet(
            in_channels=1,
            out_classes=out_classes,
            dimensions=dimensions,
            num_encoding_blocks=num_encoding_blocks,
            out_channels_first_layer=out_channels_first_layer,
            normalization=normalization,
            upsampling_type='conv',
            padding=2,
            activation='PReLU',
        )
        self.batch_size = batch_size

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)  # ??
        return optimizer

    def loss(self, logits, labels):
        loss = MultiLabelSoftMarginLoss()
        return loss(logits, labels)

    def training_step(self, batch, batch_idx):
        inputs, targets = train_batch
        logits = self.forward(inputs)
        loss = self.loss(logits, targets)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_id):
        inputs, targets = train_batch
        logits = self.forward(inputs)
        loss = self.loss(logits, targets)

        return {'loss': loss}

    def validation_end(self, outputs):
        # torch.stack: Concatenates sequence of tensors along a new dimension.
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


