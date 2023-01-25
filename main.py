import argparse
import multiprocessing
import os.path

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from lightning_lite import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.pam50.cross_validation.kfold_loop import KFoldLoop
from models.pam50.cross_validation.tiles_kfold_data_module import TilesKFoldDataModule
from data.tiles.tiles_dataset import TilesDataset
from data.tiles.tiles_labeler import TilesLabeler
from data_parser.rnr_to_metadata_parser import RNrToMetadata
from host_configuration import HostConfiguration
from models.pam50.pam50_classifier import PAM50Classifier
from preprocessor import Preprocessor


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocess", action="store_true")
    parser.add_argument("--slides_directory", type=str)
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-device", type=str)
    parser.add_argument("-tiles_dir", type=str)
    parser.add_argument("-i", "--inference", action="store_true")
    return parser


def preprocess(args):
    print("Given slides dir: ", args.slides_directory)
    preprocessor = Preprocessor(args.slides_directory)
    preprocessor.start()


def train(args):
    device = args.device
    if device is None:
        print("Device must be provided")

    tiles_directory_path = HostConfiguration.TILES_DIRECTORY.format(zoom_level=HostConfiguration.ZOOM_LEVEL,
                                                                    patch_size=HostConfiguration.PATCH_SIZE)
    if args.tiles_dir:
        tiles_directory_path = args.tiles_dir

    wandb_logger = WandbLogger(project="proteomics-project")
    num_of_workers = int(multiprocessing.cpu_count() / 2)

    print("Is going to use: {} workers".format(num_of_workers))

    rnr_to_metadata = RNrToMetadata(excel_path=HostConfiguration.RNR_METADATA_FILE_PATH)
    tiles_labeler = TilesLabeler(rnr_to_metadata)

    transform_compose = transforms.Compose([transforms.Resize(size=(299, 299)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.], std=[255.])])

    test_proportion_size = 0.15

    train_ids, test_ids = rnr_to_metadata.create_pam50_random_train_test_ids(test_size=test_proportion_size,
                                                                             tiles_directory=tiles_directory_path)

    # full_train_ids = train_ids[:]
    # train_ids, val_ids = rnr_to_metadata.split_train(full_train_ids, val_size=val_proportion_size)
    # train_dataset = TilesDataset(tiles_directory_path, transform_compose, tiles_labeler, train_ids)
    # val_dataset = TilesDataset(tiles_directory_path, transform_compose, tiles_labeler, val_ids)
    test_dataset = TilesDataset(tiles_directory_path, transform_compose, tiles_labeler, test_ids)

    model = PAM50Classifier(device).to(device)
    datamodule = TilesKFoldDataModule(tiles_directory_path, transform_compose, tiles_labeler, rnr_to_metadata,
                                      batch_size=16, num_workers=num_of_workers,
                                      test_proportion_size=test_proportion_size)
    trainer = pl.Trainer(max_epochs=2, devices="auto", accelerator="auto",
                         num_sanity_val_steps=0, logger=wandb_logger,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
                         default_root_dir=HostConfiguration.CHECKPOINTS_PATH)
    internal_fit_loop = trainer.fit_loop

    trainer.fit_loop = KFoldLoop(5, export_path=HostConfiguration.CHECKPOINTS_PATH,
                                 num_of_classes=PAM50Classifier.NUM_OF_OUT_CLASSES, device=device)
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule)

    # trainer.fit(model, train_loader, val_loader)

    # test the model
    trainer.test(model, dataloaders=DataLoader(test_dataset))


def prepare_train_env():
    if not os.path.exists(HostConfiguration.CHECKPOINTS_PATH):
        os.makedirs(HostConfiguration.CHECKPOINTS_PATH)


def inference(args):
    model = PAM50Classifier.load_from_checkpoint(args.model_path)
    softmax = nn.Softmax(dim=1)

    tiles_directory = args.tiles_dir
    ids = []
    for i in ids:
        print("Starting ", i)
        img_tiles = [t for t in os.listdir(tiles_directory) if t.startswith(i)]
        print("Found {} tiles".format(len(img_tiles)))
        if len(img_tiles) == 0:
            continue
        tensors = []
        transform_compose = transforms.Compose([transforms.Resize(size=(299, 299)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.], std=[255.])])

        for t in img_tiles:
            img = Image.open(os.path.join(tiles_directory, t))
            img = transform_compose(img)
            tensors.append(img)

        tensors = torch.stack(tensors, dim=0)
        splits = torch.tensor_split(tensors, len(tensors) // 50)
        partial_voted = []
        for split in splits:
            # batch = torch.stack(split, dim=0)
            out = model(split)
            out = softmax(out).detach().numpy()

            confidence_threshold = 0.75
            mask = out > confidence_threshold
            indexes = np.where(mask.any(axis=1), np.argmax(mask, axis=1), -1)
            tiles_values = indexes[np.where(indexes != -1)[0]]
            most_voted_tile = np.argmax(np.bincount(tiles_values.astype(int)))
            partial_voted.append(most_voted_tile)
        most_voted_tile = np.asarray(partial_voted)
        most_voted_tile = np.argmax(np.bincount(most_voted_tile.astype(int)))
        print("Id: {} got: {}".format(i, most_voted_tile))


def main():
    parser = init_argparse()
    args = parser.parse_args()

    if args.preprocess:
        preprocess(args)
    elif args.train:
        seed_everything(HostConfiguration.SEED)
        prepare_train_env()
        train(args)
    elif args.inference:
        # inference()
        pass


if __name__ == '__main__':
    main()

# python 3.9 installation: https://computingforgeeks.com/how-to-install-python-latest-debian/
# venvs guide: https://blog.eldernode.com/python-3-9-on-debian-9-and-debian-10/
# install torch for cuda 11.0 -> pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html