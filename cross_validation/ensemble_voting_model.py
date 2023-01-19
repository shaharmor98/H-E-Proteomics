from pytorch_lightning import LightningModule
import torch

from torch.nn import functional as F
from torchmetrics.classification.accuracy import Accuracy


class EnsembleVotingModel(LightningModule):
    def __init__(self, model_cls, checkpoint_paths):
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        self.test_acc = Accuracy()  # TODO- what should be accuracy params?

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # Compute the averaged predictions over the `num_folds` models.
        logits = torch.stack([m(batch[0]) for m in self.models]).mean(0)
        loss = F.nll_loss(logits, batch[1])
        self.test_acc(logits, batch[1])
        self.log("test_acc", self.test_acc)
        self.log("test_loss", loss)
