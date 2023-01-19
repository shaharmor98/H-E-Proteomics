import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from torchvision import transforms

from data.tiles.tiles_dataset import TilesDataset
from data.tiles.tiles_labeler import TilesLabeler
from data_parser.rnr_to_metadata_parser import RNrToMetadata
from host_configuration import HostConfiguration
from model import PAM50Classifier
from preprocessor import Preprocessor


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocess", action="store_true")
    parser.add_argument("--slides_directory", type=str)
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-i", "--inference", action="store_true")
    return parser


# figure out how to transfer GCP bucket into GCP VM.
# launch

def preprocess(args):
    preprocessor = Preprocessor(args.slides_directory)
    preprocessor.start()


def train():
    device = "cpu"

    rnr_to_metadata = RNrToMetadata(excel_path=HostConfiguration.RNR_METADATA_FILE_PATH)
    tiles_labeler = TilesLabeler(rnr_to_metadata)

    transform_compose = transforms.Compose([transforms.Resize(size=(299, 299)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.], std=[255.])])

    test_proportion_size = 0.1
    val_proportion_size = 0.1

    train_ids, test_ids = rnr_to_metadata.create_pam50_random_train_test_ids(test_size=test_proportion_size)
    full_train_ids = train_ids[:]
    train_ids, val_ids = rnr_to_metadata.split_train(full_train_ids, val_size=val_proportion_size)

    tiles_directory_path = HostConfiguration.TILES_DIRECTORY.format(zoom_level=HostConfiguration.ZOOM_LEVEL,
                                                                    patch_size=HostConfiguration.PATCH_SIZE)
    train_dataset = TilesDataset(tiles_directory_path, transform_compose, tiles_labeler, train_ids)
    val_dataset = TilesDataset(tiles_directory_path, transform_compose, tiles_labeler, val_ids)
    test_dataset = TilesDataset(tiles_directory_path, transform_compose, tiles_labeler, test_ids)

    model = PAM50Classifier(device).to(device)
    # devices="auto", accelerator="auto"
    trainer = pl.Trainer(max_epochs=2,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
                         default_root_dir="checkpoints")

    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=2)

    trainer.fit(model, train_loader, val_loader)

    # test the model
    trainer.test(model, dataloaders=DataLoader(test_dataset))


def inference():
    raise NotImplementedError("This feature is not implemented")


def main():
    parser = init_argparse()
    args = parser.parse_args()

    if args.preprocess:
        preprocess(args)
    elif args.train:
        train()
    elif args.inference:
        inference()


if __name__ == '__main__':
    main()
