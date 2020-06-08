import pytorch_lightning as pl
from torch.nn import MultiLabelSoftMarginLoss
from .
import torch


class LitUnet(pl.LightningModule):

    def __init__(
            self,
            initial_features: int = 32,
            depth: int = 3,
            n_labels: int = 2,
            normalization: bool = True,
            batch_size: int = 1,
    ):
        super().__init__()
        self.unet = UNet3d(
            initial_features, depth=depth, n_labels=n_labels, normalization=normalization
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


