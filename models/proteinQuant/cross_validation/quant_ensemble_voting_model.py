import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryAccuracy


class QuantEnsembleVotingModel(LightningModule):
    def __init__(self, model_cls, checkpoint_paths, device):
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        self.test_acc = BinaryAccuracy(threshold=0.75)

        self._device = device

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, original_labels = batch
        predictions = []

        for model in self.models:
            y_hat = model(x)
            predictions.append(y_hat)

        predictions = torch.stack(predictions, dim=0).numpy()
        mean_predictions = np.mean(predictions, axis=0)
        test_acc = self.test_acc(mean_predictions, original_labels)

        self.log('test_ensemble_acc', test_acc, prog_bar=True)
