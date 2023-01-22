import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.classification.accuracy import Accuracy


class EnsembleVotingModel(LightningModule):
    def __init__(self, model_cls, checkpoint_paths, num_of_classes, device):
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        self.test_acc = Accuracy(task="multiclass", num_classes=num_of_classes)
        self._num_of_classes = num_of_classes
        self._device = device

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO- revise
        x, original_labels = batch
        one_hot_labels = torch.zeros(x.shape[0], self._num_of_classes).to(self._device)
        y = one_hot_labels.scatter_(1, original_labels.view(-1, 1), 1)

        predictions = []

        for model in self.models:
            y_hat = model(x)
            predictions.append(torch.max(y_hat, dim=1))

        predictions = torch.stack(predictions, dim=0).numpy()

        # find the most frequent value in each column, that is, the majority vote of the models
        majority_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

        test_acc = torch.sum(majority_predictions == original_labels).float() / len(y)

        self.log('test_ensemble_acc', test_acc, prog_bar=True)
