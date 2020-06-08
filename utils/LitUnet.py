import pytorch_lightning as pl
from torch.nn import MultiLabelSoftMarginLoss
import torch


class LitUnet(pl.LightningModule):

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)  # ??
        return optimizer

    def loss(self, logits, labels):
        loss = MultiLabelSoftMarginLoss()
        return loss(logits, labels)

