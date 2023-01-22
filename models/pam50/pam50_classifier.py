import pytorch_lightning as pl
import torch
import torch.optim as optim
import torchmetrics
import torchvision.models as models
from torch import nn
from torchvision.models import InceptionOutputs


class PAM50Classifier(pl.LightningModule):
    NUM_OF_OUT_CLASSES = 2

    def __init__(self, device):
        super().__init__()
        self.learning_rate = 0.001
        self.model = models.inception_v3()
        self.fc = nn.Linear(1000, self.NUM_OF_OUT_CLASSES)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.NUM_OF_OUT_CLASSES)
        self._device = device
        self.save_hyperparameters()

    def forward(self, x):
        try:
            x = self.model(x)
        except Exception:
            print("Shit")
            print("Shape: ", x.shape)
            x = self.model(x)

        if isinstance(x, torch.Tensor):
            x = self.fc(x)
        elif isinstance(x, InceptionOutputs):
            x = self.fc(x[0])
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, original_labels = batch
        if len(x) == 1:
            return

        y_hat = self(x)
        one_hot_labels = torch.zeros(x.shape[0], self.NUM_OF_OUT_CLASSES).to(self._device)
        y = one_hot_labels.scatter_(1, original_labels.view(-1, 1), 1)
        loss = self.loss(y_hat, y)

        _, preds = torch.max(y_hat, dim=1)
        acc = torch.sum(preds == original_labels).float() / len(y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, original_labels = batch
        y_hat = self(x)
        one_hot_labels = torch.zeros(x.shape[0], self.NUM_OF_OUT_CLASSES).to(self._device)
        y = one_hot_labels.scatter_(1, original_labels.view(-1, 1), 1)
        val_loss = self.loss(y_hat, y)

        _, preds = torch.max(y_hat, dim=1)
        acc = torch.sum(preds == original_labels).float() / len(y)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {"loss": val_loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, original_labels = batch
        y_hat = self(x)
        one_hot_labels = torch.zeros(x.shape[0], self.NUM_OF_OUT_CLASSES).to(self._device)
        y = one_hot_labels.scatter_(1, original_labels.view(-1, 1), 1)
        test_loss = self.loss(y_hat, y)

        _, preds = torch.max(y_hat, dim=1)
        test_acc = torch.sum(preds == original_labels).float() / len(y)

        self.log('test_acc', test_acc, prog_bar=True)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
