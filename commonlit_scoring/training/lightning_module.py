import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn
from sklearn.metrics import mean_squared_error
from transformers import AutoModel


def compute_mcrmse(preds, targets):
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    rmses = [
        mean_squared_error(targets[:, i], preds[:, i]) for i in range(targets.shape[1])
    ]
    return float(np.mean(rmses))


class CommonLitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = AutoModel.from_pretrained(cfg.model.encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.regressor = nn.Linear(hidden_size, 2)
        self.loss_fn = nn.MSELoss()
        self.lr = cfg.model.lr

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.regressor(pooled)

    def training_step(self, batch, batch_idx):
        preds = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(preds, batch["targets"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(preds, batch["targets"])
        mcrmse = compute_mcrmse(preds, batch["targets"])
        self.log_dict({"val_loss": loss, "val_mcrmse": mcrmse}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
