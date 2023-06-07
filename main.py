import argparse
import json
import multiprocessing
import os.path
import random

import numpy as np
import pytorch_lightning as pl
import scipy
import torch
from PIL import Image
from lightning_lite import seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from data_parser.dia_to_metadata_parser import DiaToMetadata
from host_configuration import HostConfiguration
from models.proteinQuant.cross_validation.kfold_loop import KFoldLoop
from models.proteinQuant.cross_validation.tiles_kfold_data_module import TilesKFoldDataModule
from models.proteinQuant.protein_quant_classifier import ProteinQuantClassifier
from models.proteinQuant.tiles_dataset import TilesDataset
from preprocessor import Preprocessor
from sklearn.metrics import confusion_matrix


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
    parser.add_argument("--analysis", action="store_true")
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
    trainer = pl.Trainer(max_epochs=5, devices="auto", accelerator="auto",
                         num_sanity_val_steps=0, logger=wandb_logger, strategy="ddp",
                         default_root_dir=HostConfiguration.CHECKPOINTS_PATH.format(gene=gene))
    internal_fit_loop = trainer.fit_loop

    trainer.fit_loop = KFoldLoop(HostConfiguration.NUM_OF_FOLDS,
                                 export_path=HostConfiguration.CHECKPOINTS_PATH.format(gene=gene),
                                 device=device, current_fold=start_fold)
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule)


def prepare_train_env(gene):
    if not os.path.exists(HostConfiguration.CHECKPOINTS_PATH.format(gene=gene)):
        os.makedirs(HostConfiguration.CHECKPOINTS_PATH.format(gene=gene))


def confusion_matrix_analysis(gene, dia_metadata):
    tiles_directory = HostConfiguration.TILES_DIRECTORY.format(zoom_level=HostConfiguration.ZOOM_LEVEL,
                                                               patch_size=HostConfiguration.PATCH_SIZE)
    transform_compose = transforms.Compose([transforms.Resize(size=(299, 299)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.], std=[255.])])
    #    dia_metadata = DiaToMetadata(HostConfiguration.DIA_GENES_FILE_PATH, HostConfiguration.RNR_METADATA_FILE_PATH,
    #                                 tiles_directory)
    normalized_records = dia_metadata.get_normalized_gene_records(gene_name=gene)
    with open(HostConfiguration.PREDICTIONS_SUMMARY_FILE.format(gene=gene), 'r') as f:
        predictions = json.load(f)

    with open(HostConfiguration.TEST_IDS_FILE.format(gene=gene), 'r') as f:
        ids = json.load(f)
        test_ids = []
        for k, v in ids.items():
            test_ids.append((k, v))

    actual_prediction = []
    for slide_id, _ in test_ids:
        actual_prediction.append(normalized_records[slide_id])

    # Method B- applying threshold and then classify by majority
    preds = []
    for test_id in test_ids:
        pred = predictions[test_id[0]]
        dataset = TilesDataset(tiles_directory, transform_compose, [test_id], caller="Prediction dataset")
        threshold_predictions = np.where(np.asarray(pred) > 0.5, 1, 0)
        averaged_tile = np.mean(threshold_predictions, axis=0)
        total = np.sum(np.where(np.asarray(averaged_tile) > 0.5, 1, 0), axis=0)
        ratio = total / dataset.get_num_of_files()
        preds.append(ratio)

    upper_percentile_threshold = np.partition(actual_prediction, -5)[-5]
    binary_preds = np.where(np.asarray(preds) >= upper_percentile_threshold, 1, 0)
    binary_actual = np.where(np.asarray(actual_prediction) >= upper_percentile_threshold, 1, 0)
    cm = confusion_matrix(binary_actual, binary_preds)
    class_labels = ['low', 'high']
    cm_labeled = np.array([[f"{class_labels[i]} ({val})" for i, val in enumerate(row)] for row in cm])

    print(cm_labeled)


def inference(gene):
    checkpoint_paths = [os.path.join(HostConfiguration.CHECKPOINTS_PATH.format(gene=gene), f"model.{f_idx + 1}.pt")
                        for f_idx in range(HostConfiguration.NUM_OF_FOLDS)]
    models_paths = []
    for ckpt in checkpoint_paths:
        if os.path.exists(ckpt):
            models_paths.append(ckpt)

    tiles_directory = HostConfiguration.TILES_DIRECTORY.format(zoom_level=HostConfiguration.ZOOM_LEVEL,
                                                               patch_size=HostConfiguration.PATCH_SIZE)
    transform_compose = transforms.Compose([transforms.Resize(size=(299, 299)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.], std=[255.])])

    with open(HostConfiguration.TEST_IDS_FILE.format(gene=gene), 'r') as f:
        ids = json.load(f)
        test_ids = []
        for k, v in ids.items():
            test_ids.append((k, v))

    results = {}
    for ckpt_path in models_paths:
        model = ProteinQuantClassifier.load_from_checkpoint(ckpt_path)
        model_name = os.path.basename(ckpt_path)
        print("Starting {}".format(model_name))
        for test_id in test_ids:
            if not test_id[0] in results:
                results[test_id[0]] = []
            print("Starting test_id: ", test_id)
            dataset = TilesDataset(tiles_directory, transform_compose, [test_id], caller="Prediction dataset")
            trainer = pl.Trainer(devices=1, accelerator="auto")
            predictions = trainer.predict(model,
                                          dataloaders=DataLoader(dataset, num_workers=int(multiprocessing.cpu_count())))
            predictions = [p.item() for p in predictions]
            results[test_id[0]].append(predictions)

    with open(HostConfiguration.PREDICTIONS_SUMMARY_FILE.format(gene=gene), "w") as f:
        json.dump(results, f)


def analysis(gene):
    """
    Input- dictionary including: keys = slides ids. value- (num_of_models, H, W) of predictions
    """
    tiles_directory = HostConfiguration.TILES_DIRECTORY.format(zoom_level=HostConfiguration.ZOOM_LEVEL,
                                                               patch_size=HostConfiguration.PATCH_SIZE)
    transform_compose = transforms.Compose([transforms.Resize(size=(299, 299)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.], std=[255.])])
    dia_metadata = DiaToMetadata(HostConfiguration.DIA_GENES_FILE_PATH, HostConfiguration.RNR_METADATA_FILE_PATH,
                                 tiles_directory)
    normalized_records = dia_metadata.get_normalized_gene_records(gene_name=gene)
    A_results = []
    B_results = []
    C_results = []
    D_results = []
    E_results = []

    with open(HostConfiguration.PREDICTIONS_SUMMARY_FILE.format(gene=gene), 'r') as f:
        predictions = json.load(f)

    with open(HostConfiguration.TEST_IDS_FILE.format(gene=gene), 'r') as f:
        ids = json.load(f)
        test_ids = []
        for k, v in ids.items():
            test_ids.append((k, v))

    actual_prediction = []
    for slide_id, _ in test_ids:
        actual_prediction.append(normalized_records[slide_id])

    # Method A- averaging tile level
    for test_id in test_ids:
        pred = predictions[test_id[0]]
        dataset = TilesDataset(tiles_directory, transform_compose, [test_id], caller="Prediction dataset")
        averaged_tile = np.mean(pred, axis=0)
        total = np.sum(np.where(np.asarray(averaged_tile) > 0.5, 1, 0), axis=0)
        ratio = total / dataset.get_num_of_files()
        A_results.append(ratio)

    # Method B- applying threshold and then classify by majority
    for test_id in test_ids:
        pred = predictions[test_id[0]]
        dataset = TilesDataset(tiles_directory, transform_compose, [test_id], caller="Prediction dataset")
        threshold_predictions = np.where(np.asarray(pred) > 0.5, 1, 0)
        averaged_tile = np.mean(threshold_predictions, axis=0)
        total = np.sum(np.where(np.asarray(averaged_tile) > 0.5, 1, 0), axis=0)
        ratio = total / dataset.get_num_of_files()
        B_results.append(ratio)

    # Method C- calculate distribution of results, take the first 5 values around the mean
    for test_id in test_ids:
        pred = predictions[test_id[0]]
        pred = np.asarray(pred)
        results = np.zeros((pred.shape[1]))
        dataset = TilesDataset(tiles_directory, transform_compose, [test_id], caller="Prediction dataset")
        for i in range(pred.shape[1]):
            # Compute the mean of all values in the (i,j) index across the batches.
            mean_val = np.mean(pred[:, i])

            # Obtain the indices of the sorted values for that index across the batches
            sorted_indices = np.argsort(np.abs(pred[:, i] - mean_val))

            # Take the 5 values centered around the mean
            results[i] = np.mean(pred[sorted_indices[:5], i])

        total = np.sum(np.where(np.asarray(results) > 0.5, 1, 0), axis=0)
        ratio = total / dataset.get_num_of_files()
        C_results.append(ratio)

    # Method D- calculate distribution of results, take the first 8 values around the mean
    for test_id in test_ids:
        pred = predictions[test_id[0]]
        pred = np.asarray(pred)
        results = np.zeros((pred.shape[1]))
        dataset = TilesDataset(tiles_directory, transform_compose, [test_id], caller="Prediction dataset")
        for i in range(pred.shape[1]):
            # Compute the mean of all values in the (i,j) index across the batches.
            mean_val = np.mean(pred[:, i])

            # Obtain the indices of the sorted values for that index across the batches
            sorted_indices = np.argsort(np.abs(pred[:, i] - mean_val))

            # Take the 8 values centered around the mean
            results[i] = np.mean(pred[sorted_indices[:8], i])

        total = np.sum(np.where(np.asarray(results) > 0.5, 1, 0), axis=0)
        ratio = total / dataset.get_num_of_files()
        D_results.append(ratio)

    # Method E- averaging model's level
    for test_id in test_ids:
        pred = np.asarray(predictions[test_id[0]])
        dataset = TilesDataset(tiles_directory, transform_compose, [test_id], caller="Prediction dataset")
        model_sum = np.sum(np.where(pred > 0.5, 1, 0), axis=1)
        total = int(np.mean(model_sum))
        ratio = total / dataset.get_num_of_files()
        E_results.append(ratio)

    print("A: ", A_results)
    print("B: ", B_results)
    print("C: ", C_results)
    print("D: ", D_results)
    print("E: ", E_results)
    print("Actual prediction: ", actual_prediction)
    print("Method A: Spearman correlation for {}: {}".format(gene, scipy.stats.spearmanr(A_results, actual_prediction)))
    print("Method B: Spearman correlation for {}: {}".format(gene, scipy.stats.spearmanr(B_results, actual_prediction)))
    print("Method C: Spearman correlation for {}: {}".format(gene, scipy.stats.spearmanr(C_results, actual_prediction)))
    print("Method D: Spearman correlation for {}: {}".format(gene, scipy.stats.spearmanr(D_results, actual_prediction)))
    print("Method E: Spearman correlation for {}: {}".format(gene, scipy.stats.spearmanr(E_results, actual_prediction)))


def generate_gene_results(gene_name):
    tiles_directory = HostConfiguration.TILES_DIRECTORY.format(zoom_level=HostConfiguration.ZOOM_LEVEL,
                                                               patch_size=HostConfiguration.PATCH_SIZE)
    transform_compose = transforms.Compose([transforms.Resize(size=(299, 299)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.], std=[255.])])
    dia_metadata = DiaToMetadata(HostConfiguration.DIA_GENES_FILE_PATH, HostConfiguration.RNR_METADATA_FILE_PATH,
                                 tiles_directory)
    normalized_records = dia_metadata.get_normalized_gene_records(gene_name=gene_name)

    with open(HostConfiguration.PREDICTIONS_SUMMARY_FILE.format(gene=gene_name), 'r') as f:
        predictions = json.load(f)

    with open(HostConfiguration.TEST_IDS_FILE.format(gene=gene_name), 'r') as f:
        ids = json.load(f)
        test_ids = []
        for k, v in ids.items():
            test_ids.append((k, v))

    actual_prediction = []
    for slide_id, _ in test_ids:
        actual_prediction.append(normalized_records[slide_id])

    preds = []
    for test_id in test_ids:
        pred = predictions[test_id[0]]
        dataset = TilesDataset(tiles_directory, transform_compose, [test_id], caller="Prediction dataset")
        threshold_predictions = np.where(np.asarray(pred) > 0.5, 1, 0)
        averaged_tile = np.mean(threshold_predictions, axis=0)
        total = np.sum(np.where(np.asarray(averaged_tile) > 0.5, 1, 0), axis=0)
        ratio = total / dataset.get_num_of_files()
        preds.append(ratio)



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
        elif args.analysis:
            analysis(args.gene)


if __name__ == '__main__':
    main()

# python 3.9 installation: https://computingforgeeks.com/how-to-install-python-latest-debian/
# venvs guide: https://blog.eldernode.com/python-3-9-on-debian-9-and-debian-10/
# install torch for cuda 11.0 -> pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
