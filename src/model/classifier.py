# pytorch lightning classifier
import sys

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import TQDMProgressBar


class Classifier(pl.LightningModule):
    """
    Pl Lightning Wrapper for a classifier model. That can be used for training and validation.
    """

    def __init__(self, model, optimizer_class=torch.optim.Adam, optimizer_params={"lr": 1e-3},
                 lr_scheduler=None, lr_scheduler_params=None, lr_config=None,
                 custom_loss=None, custom_loss_on_data=False, pre_extractor=None, batch_transform=None,
                 ):
        """
        Args:
            model: The model to be trained.
            optimizer_type: Type of optimizer (e.g., 'adam', 'sgd').
            optimizer_params: Parameters for the optimizer.
            lr_scheduler: Optional learning rate scheduler.
            scheduler_params: Parameters for the learning rate scheduler.
        """
        super(Classifier, self).__init__()
        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.lr_config = lr_config
        self.loss_on_data = False if custom_loss is None else custom_loss_on_data
        self.loss = custom_loss or F.cross_entropy  # Default loss function is cross-entropy

        self.pre_extractor = pre_extractor
        self.batch_transform = batch_transform

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = self.batch_transform(batch) if self.batch_transform is not None else (x, y)
        with torch.no_grad():
            if self.pre_extractor is not None:
                x = self.pre_extractor(x)
        if self.loss_on_data:
            loss = self.loss(x, y)
        else:
            logits = self.forward(x)
            loss = self.loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            if self.pre_extractor is not None:
                x = self.pre_extractor(x)

        logits = self.forward(x)
        if self.loss_on_data:
            loss = torch.nn.functional.cross_entropy(logits, y)
        else:
            loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = self.optimizer_class(self.parameters(), **self.optimizer_params)
        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(opt, **self.lr_scheduler_params)
            if self.lr_config is not None:
                self.lr_config["scheduler"] = scheduler
            else:
                self.lr_config = {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss", "frequency": 1,
                                  "strict": True, "name": None}

            return {"optimizer": opt, "lr_scheduler": self.lr_config}
        else:
            return opt


class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar
