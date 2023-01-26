import pytorch_lightning as pl
import torch
import torch.optim as optim
import torchmetrics
import torchvision.models as models
from torch import nn
from torchvision.models import InceptionOutputs


class ProteinQuantClassifier(pl.LightningModule):
    def __init__(self, device):
        super().__init__()
        self.learning_rate = 0.001
        self.model = models.inception_v3()
        self.model.fc = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())
        self.loss = nn.BCELoss()
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
        self._device = device
        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        # if isinstance(x, torch.Tensor):
        #     x = self.fc(x)
        if isinstance(x, InceptionOutputs):
            x = x[0]
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, original_labels = batch
        original_labels = original_labels.reshape(-1, 1)
        if len(x) == 1:
            return
        y_hat = self(x)
        # one_hot_labels = torch.zeros(x.shape[0], self.NUM_OF_OUT_CLASSES).to(self._device)
        # y = one_hot_labels.scatter_(1, original_labels.view(-1, 1), 1)
        loss = self.loss(y_hat.float(), original_labels)
        accuracy = self.accuracy(y_hat, original_labels)

        # _, preds = torch.max(y_hat, dim=1)
        # acc = torch.sum(preds == original_labels).float() / len(y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy, prog_bar=True)
        return {'loss': loss, 'acc': accuracy}

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, original_labels = batch
        original_labels = original_labels.reshape(-1, 1)
        if len(x) == 1:
            return
        y_hat = self(x)
        # one_hot_labels = torch.zeros(x.shape[0], self.NUM_OF_OUT_CLASSES).to(self._device)
        # y = one_hot_labels.scatter_(1, original_labels.view(-1, 1), 1)
        val_loss = self.loss(y_hat.float(), original_labels)
        accuracy = self.accuracy(y_hat, original_labels)

        # _, preds = torch.max(y_hat, dim=1)
        # acc = torch.sum(preds == original_labels).float() / len(y)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        return {"loss": val_loss, "acc": accuracy}

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, original_labels = batch
        original_labels = original_labels.reshape(-1, 1)
        y_hat = self(x)
        # one_hot_labels = torch.zeros(x.shape[0], self.NUM_OF_OUT_CLASSES).to(self._device)
        # y = one_hot_labels.scatter_(1, original_labels.view(-1, 1), 1)
        test_loss = self.loss(y_hat.float(), original_labels)
        accuracy = self.accuracy(y_hat, original_labels)

        # _, preds = torch.max(y_hat, dim=1)
        # acc = torch.sum(preds == original_labels).float() / len(y)

        self.log('test_acc', accuracy, prog_bar=True)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
