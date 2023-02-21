import argparse
import json
import multiprocessing
import os.path

import numpy as np
import pytorch_lightning as pl
import scipy
import torch
from PIL import Image
from lightning_lite import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data_parser.dia_to_metadata_parser import DiaToMetadata
from host_configuration import HostConfiguration
from models.proteinQuant.cross_validation.kfold_loop import KFoldLoop
from models.proteinQuant.cross_validation.tiles_kfold_data_module import TilesKFoldDataModule
from models.proteinQuant.protein_quant_classifier import ProteinQuantClassifier
from models.proteinQuant.tiles_dataset import TilesDataset
from preprocessor import Preprocessor


def run(model):
    files = os.listdir("/home/shaharmor98/proteomics-tiles-shahar-mor/images/zoom_20_size_512/")
    target = "PD35985a"
    tiles_directory = "/home/shaharmor98/proteomics-tiles-shahar-mor/images/zoom_20_size_512/"
    img_tiles = [f for f in files if f.startswith(target)]
    tensors = []
    transform_compose = transforms.Compose([transforms.Resize(size=(299, 299)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.], std=[255.])])
    for t in img_tiles:
        img = Image.open(os.path.join(tiles_directory, t))
        img = transform_compose(img)
        tensors.append(img)

        tensors = torch.stack(tensors, dim=0)
        out = model.model(tensors)
        return out


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
    parser.add_argument("-i", "--inference", action="store_true")
    return parser


def preprocess(args):
    print("Given slides dir: ", args.slides_directory)
    preprocessor = Preprocessor(args.slides_directory)
    preprocessor.start()


def protein_quant_train(args, gene):
    device = args.device
    if device is None:
        print("Device must be provided")
        exit(1)

    start_fold = args.fold
    if start_fold is None:
        start_fold = 0

    tiles_directory_path = HostConfiguration.TILES_DIRECTORY.format(zoom_level=HostConfiguration.ZOOM_LEVEL,
                                                                    patch_size=HostConfiguration.PATCH_SIZE)
    if args.tiles_dir:
        tiles_directory_path = args.tiles_dir

    wandb_logger = WandbLogger(project="proteomics-project", log_model=True)
    num_of_workers = int(multiprocessing.cpu_count())

    dia_metadata = DiaToMetadata(HostConfiguration.DIA_GENES_FILE_PATH, HostConfiguration.RNR_METADATA_FILE_PATH,
                                 tiles_directory_path)
    gene_slides_with_labels = dia_metadata.get_gene_slides_with_labels(gene)
    # gene_slides_with_labels = dia_metadata.get_gene_slides_with_labels(HostConfiguration.CHOSEN_GENE)
    transform_compose = transforms.Compose([transforms.Resize(size=(299, 299)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.], std=[255.])])
    test_proportion_size = 0.15

    train, test = dia_metadata.random_shuffle(gene_slides_with_labels, test_proportion_size)
    with open(HostConfiguration.TEST_IDS_FILE.format(gene=gene), 'w') as f:
        json.dump(test, f)
    print("Test: ", test)

    model = ProteinQuantClassifier(device).to(device)
    datamodule = TilesKFoldDataModule(tiles_directory_path, transform_compose, dia_metadata, gene_slides_with_labels,
                                      batch_size=16, num_workers=num_of_workers,
                                      test_proportion_size=test_proportion_size)
    # TODO- added strategy, added pin memory to dataloader
    trainer = pl.Trainer(max_epochs=5, devices="auto", accelerator="auto",
                         num_sanity_val_steps=0, logger=wandb_logger, strategy="ddp",
                         default_root_dir=HostConfiguration.CHECKPOINTS_PATH.format(gene=gene))
    internal_fit_loop = trainer.fit_loop

    trainer.fit_loop = KFoldLoop(HostConfiguration.NUM_OF_FOLDS,
                                 export_path=HostConfiguration.CHECKPOINTS_PATH.format(gene=gene),
                                 device=device, current_fold=start_fold)
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule)


"""
def pam50_train(args):
    device = args.device
    if device is None:
        print("Device must be provided")

    tiles_directory_path = HostConfiguration.TILES_DIRECTORY.format(zoom_level=HostConfiguration.ZOOM_LEVEL,
                                                                    patch_size=HostConfiguration.PATCH_SIZE)
    if args.tiles_dir:
        tiles_directory_path = args.tiles_dir

    wandb_logger = WandbLogger(project="proteomics-project")
    num_of_workers = int(multiprocessing.cpu_count())

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

    trainer.fit_loop = KFoldLoop(HostConfiguration.NUM_OF_FOLDS, export_path=HostConfiguration.CHECKPOINTS_PATH,
                                 num_of_classes=PAM50Classifier.NUM_OF_OUT_CLASSES, device=device)
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule)

    # trainer.fit(model, train_loader, val_loader)

    # test the model
    trainer.test(model, dataloaders=DataLoader(test_dataset))
"""


def prepare_train_env(gene):
    if not os.path.exists(HostConfiguration.CHECKPOINTS_PATH.format(gene=gene)):
        os.makedirs(HostConfiguration.CHECKPOINTS_PATH.format(gene=gene))


def inference(gene):
    checkpoint_paths = [os.path.join(HostConfiguration.CHECKPOINTS_PATH.format(gene=gene), f"model.{f_idx + 1}.pt")
                        for f_idx in range(HostConfiguration.NUM_OF_FOLDS)]
    models_paths = []
    for ckpt in checkpoint_paths:
        if os.path.exists(ckpt):
            models_paths.append(ckpt)
    # models = [ProteinQuantClassifier.load_from_checkpoint(p) for p in checkpoint_paths]
    tiles_directory = HostConfiguration.TILES_DIRECTORY.format(zoom_level=HostConfiguration.ZOOM_LEVEL,
                                                               patch_size=HostConfiguration.PATCH_SIZE)
    transform_compose = transforms.Compose([transforms.Resize(size=(299, 299)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.], std=[255.])])

    # Note, those ids selected due to 42 random seed value - STAT1
    test_ids = [('PD31111a', 1), ('PD31125a', 1), ('PD31059a', 1), ('PD36036a', 1), ('PD36051a', 1),
                ('PD36019a', 0), ('PD36002a', 0), ('PD31058a', 0), ('PD36060a', 0), ('PD31098a', 0)]

    with open(HostConfiguration.TEST_IDS_FILE.format(gene=gene), 'r') as f:
        ids = json.load(f)
        test_ids = []
        for k, v in ids.items():
            test_ids.append((k, v))

    results = {}
    for ckpt_path in models_paths:
        model = ProteinQuantClassifier.load_from_checkpoint(ckpt_path)
        model_name = os.path.basename(ckpt_path)
        results[model_name] = {}
        print("Starting {}".format(model_name))
        for test_id in test_ids:
            dataset = TilesDataset(tiles_directory, transform_compose, [test_id], caller="Prediction dataset")
            trainer = pl.Trainer(devices=1, accelerator="auto")
            predictions = trainer.predict(model,
                                          dataloaders=DataLoader(dataset, num_workers=int(multiprocessing.cpu_count())))
            total = np.sum(np.where(np.asarray(predictions) > 0.5, 1, 0), axis=0)
            ratio = total / dataset.get_num_of_files()
            results[model_name][test_id[0]] = ratio
            print("ID: {} got ratio of: {}".format(test_id, ratio))
    with open(HostConfiguration.PREDICTIONS_SUMMARY_FILE.format(gene=gene), "w") as f:
        json.dump(results, f)


def spearman_correlation_test(gene):
    tiles_directory_path = HostConfiguration.TILES_DIRECTORY.format(zoom_level=HostConfiguration.ZOOM_LEVEL,
                                                                    patch_size=HostConfiguration.PATCH_SIZE)
    dia_metadata = DiaToMetadata(HostConfiguration.DIA_GENES_FILE_PATH, HostConfiguration.RNR_METADATA_FILE_PATH,
                                 tiles_directory_path)
    normalized_records = dia_metadata.get_normalized_gene_records(gene_name=gene)

    with open(HostConfiguration.PREDICTIONS_SUMMARY_FILE.format(gene=gene), "r") as f:
        results = json.load(f)
        averages = {}
        for model in results.keys():
            for sample, pred in results[model].items():
                if sample in averages:
                    averages[sample].append(pred)
                else:
                    averages[sample] = [pred]
        samples_average = {}
        for k in averages.keys():
            samples_average[k] = np.asarray(averages[k]).mean()

    # scipy.stats.spearmanr(pred, real2)
    return samples_average, normalized_records


def main():
    parser = init_argparse()
    args = parser.parse_args()

    if args.preprocess:
        preprocess(args)
    elif args.train:
        seed_everything(HostConfiguration.SEED)
        for gene in HostConfiguration.GENES:
            prepare_train_env(gene)
            protein_quant_train(args, gene)
    elif args.inference:
        inference(args.gene)


if __name__ == '__main__':
    main()

# python 3.9 installation: https://computingforgeeks.com/how-to-install-python-latest-debian/
# venvs guide: https://blog.eldernode.com/python-3-9-on-debian-9-and-debian-10/
# install torch for cuda 11.0 -> pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
