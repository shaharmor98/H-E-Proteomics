import argparse
import json
import multiprocessing
import os

import numpy as np
from lightning_lite import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from data_splitter import DataSplitter
from configuration import Configuration
from data_parser.dia_to_metadata_parser import DiaToMetadata
from models.proteinQuant.protein_quant_classifier import ProteinQuantClassifier
import pytorch_lightning as pl

from models.proteinQuant.tiles_dataset import TilesDataset

transform_compose = transforms.Compose([transforms.Resize(size=(299, 299)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.], std=[255.])])


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocess", action="store_true")
    parser.add_argument("--gene", type=str)
    parser.add_argument("--slides_directory", type=str)
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-f", "--fold", type=int)
    parser.add_argument("-device", type=str)
    parser.add_argument("-tiles_dir", type=str)
    parser.add_argument("-test_id", type=str)
    parser.add_argument("--analysis", action="store_true")
    parser.add_argument("-i", "--inference", action="store_true")
    return parser


def prepare_train_env(gene):
    if not os.path.exists(Configuration.CHECKPOINTS_PATH.format(gene=gene)):
        os.makedirs(Configuration.CHECKPOINTS_PATH.format(gene=gene))


def train(args, gene):
    device = args.device
    if device is None:
        print("Device must be provided")
        exit(1)

    tiles_directory_path = Configuration.TILES_DIRECTORY.format(zoom_level=Configuration.ZOOM_LEVEL,
                                                                patch_size=Configuration.PATCH_SIZE)
    if args.tiles_dir:
        tiles_directory_path = args.tiles_dir

    dia_metadata = DiaToMetadata(Configuration.DIA_GENES_FILE_PATH, Configuration.RNR_METADATA_FILE_PATH,
                                 tiles_directory_path)

    data_splitter = DataSplitter(dia_metadata)
    wandb_logger = WandbLogger(project="proteomics-project", log_model=True)
    num_workers = int(multiprocessing.cpu_count())

    extreme, ood = dia_metadata.split_by_expression_level(gene)

    with open(Configuration.OOD_FILE_PATH.format(gene=gene), "w") as f:
        json.dump(ood, f)

    for n_round in range(Configuration.N_ROUNDS):
        print("Starting round: " + str(n_round))
        train_instances, valid_instances = data_splitter.split_train_val(extreme, seed=Configuration.SEED + n_round)
        model = ProteinQuantClassifier(device).to(device)
        trainer = pl.Trainer(max_epochs=10, devices="auto", accelerator="auto",
                             num_sanity_val_steps=0, logger=wandb_logger, strategy="ddp",
                             callbacks=[EarlyStopping(monitor="val_epoch_loss", patience=5, mode="min")],
                             default_root_dir=Configuration.CHECKPOINTS_PATH.format(
                                 gene=gene + "-round-" + str(n_round)))
        train_dataset = TilesDataset(tiles_directory_path, transform_compose, train_instances, "Train-dataset")
        validation_dataset = TilesDataset(tiles_directory_path, transform_compose, valid_instances, "Val-dataset")

        train_loader = DataLoader(train_dataset, batch_size=Configuration.BATCH_SIZE, num_workers=num_workers,
                                  persistent_workers=True, pin_memory=True, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=Configuration.BATCH_SIZE, num_workers=num_workers,
                                       persistent_workers=True, pin_memory=True)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    run_on_ood(ood, tiles_directory_path, gene)


def run_on_ood(ood, tiles_directory, gene):
    checkpoint_path = []
    for n_round in range(Configuration.N_ROUNDS):
        model_path = "model." + gene + "-round-" + str(n_round) + ".pt"
        if os.path.exists(model_path):
            checkpoint_path.append(model_path)

    results = {}
    for ckpt_path in checkpoint_path:
        model = ProteinQuantClassifier.load_from_checkpoint(ckpt_path)
        model_name = os.path.basename(ckpt_path)
        print("Starting {}".format(model_name))
        for test_id in ood:
            if not test_id in results:
                results[test_id] = []
            print("Starting test_id: ", test_id)
            dataset = TilesDataset(tiles_directory, transform_compose, [[test_id, -1]], caller="Prediction dataset")
            trainer = pl.Trainer(devices=1, accelerator="auto")
            predictions = trainer.predict(model,
                                          dataloaders=DataLoader(dataset, num_workers=int(multiprocessing.cpu_count())))
            predictions = [p.item() for p in predictions]
            results[test_id].append(predictions)

    with open(Configuration.PREDICTIONS_SUMMARY_FILE.format(gene=gene), "w") as f:
        json.dump(results, f)


def analyse_results(gene):
    tiles_directory = Configuration.TILES_DIRECTORY.format(zoom_level=Configuration.ZOOM_LEVEL,
                                                           patch_size=Configuration.PATCH_SIZE)
    dia_metadata = DiaToMetadata(Configuration.DIA_GENES_FILE_PATH, Configuration.RNR_METADATA_FILE_PATH,
                                 tiles_directory)
    normalized_records = dia_metadata.get_normalized_gene_records(gene_name=gene)
    with open(Configuration.PREDICTIONS_SUMMARY_FILE.format(gene=gene), 'r') as f:
        predictions = json.load(f)

    with open(Configuration.OOD_FILE_PATH.format(gene), 'r') as f:
        ood = json.load(f)

    actual_prediction = []
    for slide_id in ood:
        actual_prediction.append(normalized_records[slide_id])

    pred_expression_level = []
    for test_id in ood:
        pred = predictions[test_id]
        dataset = TilesDataset(tiles_directory, transform_compose, [[test_id, -1]], caller="Prediction dataset")
        total = np.sum(np.where(np.asarray(pred) > 0.5, 1, 0))
        ratio = total / dataset.get_num_of_files()
        pred_expression_level.append(ratio)

    return actual_prediction, pred_expression_level


def main():
    parser = init_argparse()
    args = parser.parse_args()

    if args.train:
        print("Yallha")
        seed_everything(Configuration.SEED)
        for gene in Configuration.GENES:
            prepare_train_env(gene)
            train(args, gene)


if __name__ == '__main__':
    main()
