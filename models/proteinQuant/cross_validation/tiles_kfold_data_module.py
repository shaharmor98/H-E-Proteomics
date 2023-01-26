from abc import ABC, abstractmethod

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from models.proteinQuant.tiles_dataset import TilesDataset


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


class TilesKFoldDataModule(BaseKFoldDataModule):
    def __init__(self, tiles_directory, transform, dia_metadata, gene_slides_with_labels,
                 batch_size, num_workers, test_proportion_size=0.1):
        super().__init__()
        self.tiles_directory = tiles_directory
        self.transform = transform
        self.dia_metadata = dia_metadata
        self.gene_slides_with_labels = gene_slides_with_labels
        self.test_proportion_size = test_proportion_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_indices = None
        self.test_indices = None
        self.num_folds = None
        self.splits = None
        self.train_fold = None
        self.val_fold = None

    def setup(self, stage):
        self.train_indices, self.test_indices = self.dia_metadata.random_shuffle(self.gene_slides_with_labels)
        print("Those are kfold test indices: ", self.test_indices)

    def setup_folds(self, num_folds):
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_indices)))]

    def setup_fold_index(self, fold_index):
        train_indices, val_indices = self.splits[fold_index]
        train_indices, val_indices = self._translate_indices(train_indices, val_indices)
        self.train_fold = TilesDataset(self.tiles_directory, self.transform, train_indices)
        self.val_fold = TilesDataset(self.tiles_directory, self.transform, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_fold, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_fold, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(TilesDataset(self.tiles_directory, self.transform, self.test_indices))

    def __post_init__(cls):
        super().__init__()

    def _translate_indices(self, train_indices, val_indices):
        # KFold split by index, thus we will translate back to SCANB_PD_ID indices
        translated_train_indices = []
        translated_val_indices = []

        for t in train_indices:
            translated_train_indices.append(self.train_indices[t])

        for t in val_indices:
            translated_val_indices.append(self.train_indices[t])

        return translated_train_indices, translated_val_indices
