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

import json
import multiprocessing
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from host_configuration import HostConfiguration
from models.pytorch.morpohlogical_extractor import MorphologicalFeatureExtractor
# from models.pytorch.protein_quant_predictor import ProteinQuantPredictor
# from models.pytorch.texture_extractor import TextureFeaturesExtractor
# from models.pytorch.tiles.tiles_dataset import TilesDataset
from torchvision import transforms
import wandb
from tqdm import tqdm


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


"""
Method A:  NFKB2: correlation=-0.07878787878787878, pvalue=0.8287173946974606)
Method B:  NFKB2: correlation=-0.07878787878787878, pvalue=0.8287173946974606)
Method C:  NFKB2: correlation=-0.07878787878787878, pvalue=0.8287173946974606)
Method D:  NFKB2: correlation=-0.07878787878787878, pvalue=0.8287173946974606)
Method E:  NFKB2: correlation=-0.07878787878787878, pvalue=0.8287173946974606)

Method A:  EIF2B3: correlation=0.24848484848484845, pvalue=0.48877630451924314)
Method B:  EIF2B3: correlation=0.24848484848484845, pvalue=0.48877630451924314)
Method C:  EIF2B3: correlation=0.24848484848484845, pvalue=0.48877630451924314)
Method D:  EIF2B3: correlation=0.24848484848484845, pvalue=0.48877630451924314)
Method E:  EIF2B3: correlation=0.17575757575757575, pvalue=0.6271883447764844)

Method A:  HLA-C: correlation=0.7333333333333332, pvalue=0.01580059625057158)
Method B:  HLA-C: correlation=0.6606060606060605, pvalue=0.03758837757140939)
Method C:  HLA-C: correlation=0.7333333333333332, pvalue=0.01580059625057158)
Method D:  HLA-C: correlation=0.6606060606060605, pvalue=0.03758837757140939)
Method E:  HLA-C: correlation=0.7212121212121211, pvalue=0.018573155089460208)

Method A:  STAT1: correlation=0.8666666666666665, pvalue=0.0011735381801554687)
Method B:  STAT1: correlation=0.8666666666666665, pvalue=0.0011735381801554687)
Method C:  STAT1: correlation=0.8666666666666665, pvalue=0.0011735381801554687)
Method D:  STAT1: correlation=0.8666666666666665, pvalue=0.0011735381801554687)
Method E:  STAT1: correlation=0.8303030303030302, pvalue=0.0029402270232795065)

Method A:  CC2D1A: correlation=0.23636363636363633, pvalue=0.5108853175152002)
Method B:  CC2D1A: correlation=0.23636363636363633, pvalue=0.5108853175152002)
Method C:  CC2D1A: correlation=0.23636363636363633, pvalue=0.5108853175152002)
Method D:  CC2D1A: correlation=0.23636363636363633, pvalue=0.5108853175152002)
Method E:  CC2D1A: correlation=0.22424242424242422, pvalue=0.5334005612725947)

Method A:  IL1RL1: correlation=0.38181818181818183, pvalue=0.27625533338543595)
Method B:  IL1RL1: correlation=0.4424242424242424, pvalue=0.20042268671194224)
Method C:  IL1RL1: correlation=0.4424242424242424, pvalue=0.20042268671194224)
Method D:  IL1RL1: correlation=0.4424242424242424, pvalue=0.20042268671194224)
Method E:  IL1RL1: correlation=0.43030303030303024, pvalue=0.21449233253280775)

Method A:  MAPK13: correlation=0.19999999999999998, pvalue=0.5795840000000001)
Method B:  MAPK13: correlation=0.19999999999999998, pvalue=0.5795840000000001)
Method C:  MAPK13: correlation=0.19999999999999998, pvalue=0.5795840000000001)
Method D:  MAPK13: correlation=0.19999999999999998, pvalue=0.5795840000000001)
Method E:  MAPK13: correlation=0.23636363636363633, pvalue=0.5108853175152002)

Method A:  ATP5MK: correlation=0.11515151515151514, pvalue=0.7514196523258483)
Method B:  ATP5MK: correlation=0.12727272727272726, pvalue=0.7260570147627894)
Method C:  ATP5MK: correlation=0.11515151515151514, pvalue=0.7514196523258483)
Method D:  ATP5MK: correlation=0.11515151515151514, pvalue=0.7514196523258483)
Method E:  ATP5MK: correlation=0.2121212121212121, pvalue=0.5563057751029299)

Method A: HDAC1: correlation=0.7696969696969697, pvalue=0.009221952722215994)
Method B: HDAC1: correlation=0.7696969696969697, pvalue=0.009221952722215994)
Method C: HDAC1: correlation=0.7696969696969697, pvalue=0.009221952722215994)
Method D: HDAC1: correlation=0.7696969696969697, pvalue=0.009221952722215994)
Method E: HDAC1: correlation=0.7696969696969697, pvalue=0.009221952722215994)
"""


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

    prediction = []
    real = []
    for k in samples_average:
        prediction.append(samples_average[k])
        real.append(normalized_records[k])

    print("Spearman correlation for {}: {}".format(gene, scipy.stats.spearmanr(prediction, real)))

    # Spearman correlation for STAT1: SpearmanResult(correlation=0.8787878787878788, pvalue=0.0008138621117322101)
    # Spearman correlation for HLA-C: SpearmanrResult(correlation=0.7212121212121211, pvalue=0.018573155089460208)

    return samples_average, normalized_records


def get_random_split(dataset, proportion):
    """
    Input expected to be a dict

    255 elements.
    for every percentile- 3 variables
    """
    values = np.asarray(list(dataset.values()))
    train_set = {}
    test_set = {}

    def items_within(low_percentile, values, dataset):
        low = np.percentile(values, low_percentile * 10)
        high = np.percentile(values, (low_percentile + 1) * 10)
        range_values = {k: v for k, v in dataset.items() if (v >= low) and (v <= high)}
        keys = list(range_values.keys())
        random.shuffle(keys)
        split_idx = int(len(keys) * proportion) + 1
        test_keys = keys[:split_idx]
        train_keys = keys[split_idx:]
        return {k: dataset[k] for k in train_keys}, {k: dataset[k] for k in test_keys}

    for i in range(10):
        intermediate_train, intermediate_test = items_within(i, values, dataset)
        train_set.update(intermediate_train)
        test_set.update(intermediate_test)
    return train_set, test_set


def eval_model(gene):
    """
    tiles_directory = HostConfiguration.TILES_DIRECTORY.format(zoom_level=HostConfiguration.ZOOM_LEVEL,
                                                               patch_size=HostConfiguration.PATCH_SIZE)

    with open(HostConfiguration.TEST_IDS_FILE.format(gene=gene), 'r') as f:
        ids = json.load(f)
        test_ids = []
        for k, v in ids.items():
            test_ids.append({k: v})

    transform_compose = transforms.Compose([
        # transform_compose = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.], std=[255.])])
    gray_to_rgb_transforms = transforms.Compose([
        transforms.ToPILImage(),  # convert tensor to PIL Image
        transforms.Grayscale(num_output_channels=3),  # convert grayscale to RGB
        transforms.ToTensor(),  # convert PIL Image to tensor
    ])
    results = {}
    ckpt_path = "/home/shaharmor98/checkpoints/hybrid/STAT1/model-erosion-1000-outputs-5-epochs.ckpt"
    model = ProteinQuantPredictor.load_from_checkpoint(ckpt_path)
    model_name = os.path.basename(ckpt_path)
    print("Starting {}".format(model_name))
    trainer = pl.Trainer(devices=1, accelerator="auto")
    for i, test_id in enumerate(test_ids):
        key_name = list(test_id.keys())[0]
        if not key_name in results:
            results[key_name] = []
        print("{}: Starting test_id: {}".format(i, test_id))
        dataset = TilesDataset(tiles_directory, transform_compose, gray_to_rgb_transforms, test_id)
        predictions = trainer.predict(model,
                                      dataloaders=DataLoader(dataset, num_workers=int(multiprocessing.cpu_count()),
                                                             pin_memory=True, persistent_workers=True))
        predictions = [p.item() for p in predictions]
        results[key_name].append(predictions)

    actual = []
    pred = []

    for t in test_ids:
        k = list(t.keys())[0]
        actual.append(t[k])
        pred.append(np.mean(results[k]))
        # scipy.stats.spearmanr(pred, actual)
    return results
    """

def train(gene):
    """
    wandb_logger = WandbLogger(project="proteomics-project", log_model=True)
    device = "cuda"

    seed_everything(42, workers=True)
    random.seed(42)

    random_image = np.random.rand(224, 224, 3)
    textures_features = TextureFeaturesExtractor().extract(random_image)

    print("textures_features.shape[0]: {}".format(textures_features.shape[0]))
    model = ProteinQuantPredictor(textures_features.shape[0], device)

    tiles_directory_path = HostConfiguration.TILES_DIRECTORY.format(zoom_level=HostConfiguration.ZOOM_LEVEL,
                                                                    patch_size=HostConfiguration.PATCH_SIZE)

    num_of_workers = int(multiprocessing.cpu_count())

    dia_metadata = DiaToMetadata(HostConfiguration.DIA_GENES_FILE_PATH, HostConfiguration.RNR_METADATA_FILE_PATH,
                                 tiles_directory_path)
    gene_slides_with_labels = dia_metadata.get_continuous_normalized_records(gene)

    transform_compose = transforms.Compose([
        # transform_compose = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.], std=[255.])])
    gray_to_rgb_transforms = transforms.Compose([
        transforms.ToPILImage(),  # convert tensor to PIL Image
        transforms.Grayscale(num_output_channels=3),  # convert grayscale to RGB
        transforms.ToTensor(),  # convert PIL Image to tensor
    ])

    test_proportion_size = 0.15
    val_proportion_size = 0.15

    print("Starting with total: {} values".format(len(gene_slides_with_labels)))
    train_set, test_set = get_random_split(gene_slides_with_labels, test_proportion_size)
    print("Train set: {}, Test set: {} ".format(len(train_set), len(test_set)))
    train_set, val_set = get_random_split(train_set, val_proportion_size)
    print("Train set: {}, Val set: {} ".format(len(train_set), len(val_set)))

    with open(HostConfiguration.TEST_IDS_FILE.format(gene=gene), 'w') as f:
        json.dump(test_set, f)
    print("Test: ", test_set)

    train_dataset = TilesDataset(tiles_directory_path, transform_compose, gray_to_rgb_transforms, train_set)
    val_dataset = TilesDataset(tiles_directory_path, transform_compose, gray_to_rgb_transforms, val_set)
    test_dataset = TilesDataset(tiles_directory_path, transform_compose, gray_to_rgb_transforms, test_set)
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=num_of_workers,
                              persistent_workers=True, pin_memory=True, shuffle=True,
                              prefetch_factor=64)  # , prefetch_factor=64)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=num_of_workers,
                            persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(train_dataset, batch_size=16, num_workers=num_of_workers,
                             persistent_workers=True, pin_memory=True)

    model = model.to(device)
    trainer = pl.Trainer(max_epochs=5, devices="auto", accelerator="auto",
                         num_sanity_val_steps=0, logger=wandb_logger, strategy="ddp",
                         default_root_dir=HostConfiguration.CHECKPOINTS_PATH.format(gene=gene))
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(
        os.path.join(HostConfiguration.CHECKPOINTS_PATH.format(gene=gene), "model-erosion-1000-outputs-5-epochs.ckpt"))
    """


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
