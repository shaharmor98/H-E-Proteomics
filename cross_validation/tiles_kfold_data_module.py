from abc import ABC, abstractmethod
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from pytorch_lightning import LightningDataModule

from data.tiles.tiles_dataset import TilesDataset


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


class TilesKFoldDataModule(BaseKFoldDataModule):
    def __init__(self, tiles_directory, transform, tiles_labeler, rnr_to_metadata, batch_size, num_workers,
                 test_proportion_size=0.1):
        super().__init__()
        self.tiles_directory = tiles_directory
        self.transform = transform
        self.tiles_labeler = tiles_labeler
        self.rnr_to_metadata = rnr_to_metadata
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
        self.train_indices, self.test_indices = \
            self.rnr_to_metadata.create_pam50_random_train_test_ids(test_size=self.test_proportion_size)

    def setup_folds(self, num_folds):
        self.num_folds = num_folds
        print("Setup folds")
        print("There are {} train indices".format(len(self.train_indices)))
        print("Example 10 indices: {} ".format(self.train_indices[0:10]))
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_indices)))]

    def setup_fold_index(self, fold_index):
        train_indices, val_indices = self.splits[fold_index]
        print("setup fold index: ", fold_index)
        print("setup fold index: {} found {} train indices".format(fold_index, len(train_indices)))
        print("Example 10 indices: {} ".format(train_indices[0:10]))
        self.train_fold = TilesDataset(self.tiles_directory, self.transform, self.tiles_labeler, train_indices)
        self.val_fold = TilesDataset(self.tiles_directory, self.transform, self.tiles_labeler, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_fold, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_fold, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(TilesDataset(self.tiles_directory, self.transform, self.tiles_labeler, self.test_indices))

    def __post_init__(cls):
        super().__init__()
