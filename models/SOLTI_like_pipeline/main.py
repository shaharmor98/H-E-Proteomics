import argparse
import glob
import itertools
import json
import multiprocessing
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr

from sklearn import metrics as metrc
import pandas as pd
import pytorch_lightning as pl
from lightning_lite import seed_everything
from matplotlib import pyplot
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms

from configuration import Configuration
from data_parser.dia_to_metadata_parser import DiaToMetadata
from data_splitter import DataSplitter
from models.proteinQuant.protein_quant_classifier import ProteinQuantClassifier
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

    if args.tiles_dir:
        tiles_directory_path = args.tiles_dir

    else:
        tiles_directory_path = Configuration.TILES_DIRECTORY.format(zoom_level=Configuration.ZOOM_LEVEL,
                                                                    patch_size=Configuration.PATCH_SIZE)

    dia_metadata = DiaToMetadata(Configuration.DIA_GENES_FILE_PATH, Configuration.RNR_METADATA_FILE_PATH,
                                 tiles_directory_path)

    data_splitter = DataSplitter(dia_metadata)
    num_workers = int(multiprocessing.cpu_count())

    extreme, ood = dia_metadata.split_by_expression_level(gene)

    with open(Configuration.OOD_FILE_PATH.format(gene=gene), "w") as f:
        json.dump(ood, f)
    project_name = "proteomics-project"

    versions = []
    for path in glob.glob(Configuration.CHECKPOINTS_PATH.format(gene=gene) + "/" + project_name + "/*--v_*"):
        name = path.split("/")[-1]
        version = int(name.split("_")[-1])
        versions.append(version)

    if len(versions) == 0:
        version = 0
    else:
        version = max(versions) + 1

    run_version = "{gene}--v_{version}".format(gene=gene, version=str(version))
    wandb_logger = WandbLogger(project=project_name, log_model=True,
                               save_dir=Configuration.CHECKPOINTS_PATH.format(gene=gene),
                               version=run_version)
    for n_round in range(Configuration.N_ROUNDS):
        print("Starting round: " + str(n_round))
        train_instances, valid_instances = data_splitter.split_train_val(extreme,
                                                                         seed=Configuration.SEED + n_round,
                                                                         val_proportion=0.35)
        with open(Configuration.VAL_FILE_PATH.format(gene=gene, n_round=n_round), "w") as f:
            json.dump(valid_instances, f)
        model = ProteinQuantClassifier(device).to(device)
        trainer = pl.Trainer(max_epochs=10, devices="auto", accelerator="auto",
                             num_sanity_val_steps=0, logger=wandb_logger, strategy="ddp",
                             callbacks=[EarlyStopping(monitor="val_epoch_loss", patience=5, mode="min")])
        trainer.checkpoint_callback.filename = "gene-" + gene + "-round-" + str(n_round)
        train_dataset = TilesDataset(tiles_directory_path, transform_compose, train_instances, "Train-dataset")
        validation_dataset = TilesDataset(tiles_directory_path, transform_compose, valid_instances, "Val-dataset")

        train_loader = DataLoader(train_dataset, batch_size=Configuration.BATCH_SIZE, num_workers=num_workers,
                                  persistent_workers=True, pin_memory=True, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=Configuration.BATCH_SIZE, num_workers=num_workers,
                                       persistent_workers=True, pin_memory=True)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    run_on_ood(ood, tiles_directory_path, gene, run_version, project_name, "ood-predictions.txt")
    for n_round in range(Configuration.N_ROUNDS):
        with open(Configuration.VAL_FILE_PATH.format(gene=gene, n_round=n_round), "r") as f:
            valid_instances = json.load(f)
            valid_instances = [v[0] for v in valid_instances]
        run_on_ood(valid_instances, tiles_directory_path, gene, run_version, project_name,
                   "val-predictions-round-" + str(n_round) + ".txt")


def run_on_ood(ids, tiles_directory, gene, run_version, project_name, target_file_name):
    checkpoint_path = []
    checkpoints_dir = os.path.join(Configuration.CHECKPOINTS_PATH.format(gene=gene), project_name, run_version,
                                   "checkpoints")
    for n_round in range(Configuration.N_ROUNDS):
        model_path = os.path.join(checkpoints_dir, "gene-" + gene + "-round-" + str(n_round) + "-v1.ckpt")
        if os.path.exists(model_path):
            checkpoint_path.append(model_path)
    results = {}
    for ckpt_path in checkpoint_path:
        # for ckpt_path in checkpoint_path:
        model = ProteinQuantClassifier.load_from_checkpoint(ckpt_path)
        model_name = os.path.basename(ckpt_path)
        print("Starting {}".format(model_name))
        for test_id in ids:
            if not test_id in results:
                results[test_id] = []
            print("Starting test_id: ", test_id)
            dataset = TilesDataset(tiles_directory, transform_compose, [[test_id, -1]], caller="Prediction dataset")
            trainer = pl.Trainer(devices=1, accelerator="auto")
            predictions = trainer.predict(model,
                                          dataloaders=DataLoader(dataset, num_workers=int(multiprocessing.cpu_count())))
            predictions = [p.item() for p in predictions]
            results[test_id].append(predictions)

    with open(os.path.join(checkpoints_dir, target_file_name), "w") as f:
        json.dump(results, f)


def analyse_results(gene):
    tiles_directory = Configuration.TILES_DIRECTORY.format(zoom_level=Configuration.ZOOM_LEVEL,
                                                           patch_size=Configuration.PATCH_SIZE)
    dia_metadata = DiaToMetadata(Configuration.DIA_GENES_FILE_PATH, Configuration.RNR_METADATA_FILE_PATH,
                                 tiles_directory)
    normalized_records = dia_metadata.get_continuous_normalized_records(gene_name=gene)
    with open(Configuration.PREDICTIONS_SUMMARY_FILE.format(gene=gene), 'r') as f:
        predictions = json.load(f)

    with open(Configuration.OOD_FILE_PATH.format(gene=gene), 'r') as f:
        ood = json.load(f)

    actual_prediction = []
    for slide_id in ood:
        actual_prediction.append(normalized_records[slide_id])

    pred_binary_level = []
    for test_id in ood:
        pred = predictions[test_id]
        pred = np.asarray(pred)
        pred = np.where(pred > 0.5, 1, 0)
        models_predictions = np.mean(pred, axis=1)
        models_predictions = np.where(models_predictions > 0.5, 1, 0)
        final_prediction = int(np.mean(models_predictions) > 0.5)
        pred_binary_level.append(final_prediction)

    def get_score_prediction_per_model(ood, predictions):
        res = []
        for i in range(Configuration.N_ROUNDS):
            models_res = []
            for test_id in ood:
                pred = predictions[test_id]
                pred = np.asarray(pred)[i, :]
                pred = np.where(pred > 0.5, 1, 0)
                prediction = np.mean(pred)
                # models_predictions = np.where(models_predictions > 0.5, 1, 0)
                # final_prediction = int(models_predictions > 0.5)
                models_res.append(prediction)
            res.append(models_res)
        return res

    def per_model_auc(actual, scores):
        aucs = []
        for i in range(Configuration.N_ROUNDS):
            aucs.append(roc_auc_score(actual, scores[i]))
        return sorted(aucs)

    pred_scores = []
    for test_id in ood:
        pred = predictions[test_id]
        pred = np.asarray(pred)
        # dataset = TilesDataset(tiles_directory, transform_compose, [[test_id, -1]], caller="Prediction dataset")
        pred = np.where(pred > 0.5, 1, 0)
        models_predictions = np.mean(pred, axis=1)
        pred_scores.append(np.mean(models_predictions))

    median = np.percentile(np.asarray(list(normalized_records.values())), 50)
    true_labels = []
    for test_id in ood:
        true_labels.append(int(normalized_records[test_id] > median))

    # create pandas dataframe from the results
    df = pd.DataFrame({'true_label': true_labels, 'pred': pred_binary_level, 'score': pred_scores})

    # calculate spearman and pearson correlation
    spearman_corr = spearmanr(df['true_label'], df['score'])
    pearson_corr = pearsonr(df['true_label'], df['score'])
    spearman_corr = spearmanr(actual_prediction, df['score'])
    pearson_corr = pearsonr(actual_prediction, df['score'])


def smaller_ood_fraction(gene_name):
    tiles_directory = Configuration.TILES_DIRECTORY.format(zoom_level=Configuration.ZOOM_LEVEL,
                                                           patch_size=Configuration.PATCH_SIZE)
    dia_metadata = DiaToMetadata(Configuration.DIA_GENES_FILE_PATH, Configuration.RNR_METADATA_FILE_PATH,
                                 tiles_directory)

    row = dia_metadata.get_gene_normalized_quant(gene_name)
    rnrs = dia_metadata.get_rnrs()
    ood_near_high_low_percentile = np.percentile(row, 60)
    ood_near_high_high_percentile = np.percentile(row, 70)
    ood_near_low_high_percentile = np.percentile(row, 40)
    ood_near_low_low_percentile = np.percentile(row, 30)

    ood_near_high_cols = row[(row > ood_near_high_low_percentile) & (row < ood_near_high_high_percentile)].dropna(
        axis=1).columns
    ood_near_low_cols = row[(row > ood_near_low_low_percentile) & (row < ood_near_low_high_percentile)].dropna(
        axis=1).columns

    ood_rnrs = [record[record.find("_") + 1:] for record in ood_near_high_cols.to_list() if
                record.startswith('ProteinQuant_')]
    ood_rnrs.extend([record[record.find("_") + 1:] for record in ood_near_low_cols.to_list() if
                     record.startswith('ProteinQuant_')])

    ood_ids = []
    for rnr in ood_rnrs:
        ood_ids.append(list(filter(lambda x: rnrs[x] == rnr, rnrs))[0])

    normalized_records = dia_metadata.get_continuous_normalized_records(gene_name=gene_name)
    median = np.percentile(np.asarray(list(normalized_records.values())), 50)
    true_labels = []
    for test_id in ood_ids:
        true_labels.append(int(normalized_records[test_id] > median))

    with open(Configuration.PREDICTIONS_SUMMARY_FILE.format(gene=gene_name), 'r') as f:
        predictions = json.load(f)

    pred_binary_level = []
    for test_id in ood_ids:
        pred = predictions[test_id]
        pred = np.asarray(pred)
        pred = np.where(pred > 0.5, 1, 0)
        models_predictions = np.mean(pred, axis=1)
        models_predictions = np.where(models_predictions > 0.5, 1, 0)
        final_prediction = int(np.mean(models_predictions) > 0.5)
        pred_binary_level.append(final_prediction)

    pred_scores = []
    for test_id in ood_ids:
        pred = predictions[test_id]
        pred = np.asarray(pred)
        pred = np.where(pred > 0.5, 1, 0)
        models_predictions = np.mean(pred, axis=1)
        pred_scores.append(np.mean(models_predictions))


def create_PR_curve(df_multi_model_res, output_path):
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    lr_precision, lr_recall, _ = precision_recall_curve(df_multi_model_res['true_label'], df_multi_model_res['score'])
    lr_f1, lr_auc = f1_score(df_multi_model_res['true_label'], df_multi_model_res['pred']), auc(lr_recall, lr_precision)
    # plot the precision-recall curves
    no_skill = len(df_multi_model_res['true_label'][df_multi_model_res['true_label'] == 1]) / len(
        df_multi_model_res['true_label'])
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Defualt', c='#0f0f0f')
    plt.plot(lr_recall, lr_precision, label='Classifier')
    # axis labels
    plt.xlabel('Recall', size=14)
    plt.ylabel('Precision', size=14)
    axes.set_title('PR Curve, F1_score=%.3f' % (lr_f1), size=16)
    # show the legend
    plt.legend()
    # show the plot
    # plt.show()
    plt.savefig(output_path)
    plt.close(fig)


def create_ROC_curve(df_multi_model_res, output_path):
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    testy = df_multi_model_res['true_label']
    ns_probs = np.zeros(len(df_multi_model_res))
    # predict probabilities
    lr_probs = df_multi_model_res['score']
    # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)
    # summarize scores
    print('Defualt: ROC AUC=%.3f' % (ns_auc))
    print('Classifier: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Defualt', c='#0f0f0f')
    pyplot.plot(lr_fpr, lr_tpr, label='Classifier')
    # axis labels
    pyplot.xlabel('False Positive Rate', size=14)
    pyplot.ylabel('True Positive Rate', size=14)
    axes.set_title('ROC Curve, AUC=%.3f' % (lr_auc), size=16)
    # show the legend
    pyplot.legend()
    # show the plot
    # pyplot.show()
    plt.savefig(output_path)
    plt.close(fig)


def create_df_confusion_matrix(y_true, y_pred, normalize=None):
    res_conf_matrix = metrc.confusion_matrix(y_true, y_pred, normalize=normalize, labels=[True, False]).T
    rows = np.expand_dims(np.array(['Predicted Positive', 'Predicted Negative']), axis=1)
    res_conf_matrix_with_rows = np.concatenate((rows, res_conf_matrix), axis=1)
    return pd.DataFrame(res_conf_matrix_with_rows, columns=['', 'Actual Positive', 'Actual Negative'])


def get_score_prediction_per_model(ood, predictions):
    res = []
    for i in range(Configuration.N_ROUNDS):
        models_res = []
        for test_id in ood:
            pred = predictions[test_id]
            pred = np.asarray(pred)[i, :]
            pred = np.where(pred > 0.5, 1, 0)
            prediction = np.mean(pred)
            # models_predictions = np.where(models_predictions > 0.5, 1, 0)
            # final_prediction = int(models_predictions > 0.5)
            models_res.append(prediction)
        res.append(models_res)
    return res


def per_model_auc(actual, scores):
    aucs = []
    for i in range(Configuration.N_ROUNDS):
        aucs.append(roc_auc_score(actual, scores[i]))
    # return sorted(aucs)
    return aucs


def final_results_method():
    dia_metadata = DiaToMetadata(Configuration.DIA_GENES_FILE_PATH, Configuration.RNR_METADATA_FILE_PATH, "")
    with open("/Users/shahar.mor/HE_Results/aug_27/MKI67/normalized_mki67.txt", 'r') as f:
        normalized_records = json.load(f)
    with open("/Users/shahar.mor/HE_Results/aug_27/MKI67/ood.txt", 'r') as f:
        ood = json.load(f)
    with open("/Users/shahar.mor/HE_Results/aug_27/MKI67/ood-predictions.txt", 'r') as f:
        predictions = json.load(f)

    median = np.percentile(np.asarray(list(normalized_records.values())), 50)
    true_labels = []
    for test_id in ood:
        true_labels.append(int(normalized_records[test_id] > median))

    # per model results
    res = get_score_prediction_per_model(ood, predictions)
    aucs = per_model_auc(true_labels, res)

    # mean results
    pred_binary_level = []
    for test_id in ood:
        pred = predictions[test_id]
        pred = np.asarray(pred)
        pred = np.where(pred > 0.5, 1, 0)
        models_predictions = np.mean(pred, axis=1)
        models_predictions = np.where(models_predictions > 0.5, 1, 0)
        final_prediction = np.mean(models_predictions)
        # final_prediction = int(np.mean(models_predictions) > 0.5)
        pred_binary_level.append(final_prediction)
    auc = roc_auc_score(true_labels, pred_binary_level)

    # find val aucs, pick top 3, then find ood aucs
    val_ids = []
    for n_round in range(Configuration.N_ROUNDS):
        with open("/Users/shahar.mor/HE_Results/aug_27/MKI67/val_{}.txt".format(str(n_round)), 'r') as f:
            val_ids.append(json.load(f))
            # val_ids[-1] = [x[0] for x in val_ids[-1]]

    val_predictions = []
    for n_round in range(Configuration.N_ROUNDS):
        with open("/Users/shahar.mor/HE_Results/aug_27/MKI67/val-predictions-round-{}.txt".format(str(n_round)),
                  'r') as f:
            val_predictions.append(json.load(f))

    # find val aucs
    val_aucs = []
    for n_round in range(Configuration.N_ROUNDS):
        res = []
        true_labels = []
        for test_id, label in val_ids[n_round]:
            pred = val_predictions[n_round][test_id]
            pred = np.asarray(pred)
            pred = np.where(pred > 0.5, 1, 0)
            prediction = np.mean(pred)
            res.append(prediction)
            true_labels.append(label)

        val_aucs.append(roc_auc_score(true_labels, res))

    # after manual inspection, choose: 0, 1, 4
    true_labels = []
    for test_id in ood:
        true_labels.append(int(normalized_records[test_id] > median))

    indices = [0, 1, 4]
    # indices = [0, 2, 3]
    pred_binary_level = []
    for test_id in ood:
        pred = predictions[test_id]
        pred = np.asarray(pred)[indices, :]
        pred = np.where(pred > 0.5, 1, 0)
        models_predictions = np.mean(pred, axis=1)
        # models_predictions = np.where(models_predictions > 0.5, 1, 0)
        final_prediction = np.mean(models_predictions)
        pred_binary_level.append(final_prediction)
    auc = roc_auc_score(true_labels, pred_binary_level)

    # ood near
    ood_near = []
    values = list(normalized_records.values())
    lower_percentile = np.percentile(values, 60)
    upper_percentile = np.percentile(values, 70)
    for name, val in normalized_records.items():
        if lower_percentile <= val <= upper_percentile:
            ood_near.append(name)
            # ood_near.append((name, 1))

    lower_percentile = np.percentile(values, 30)
    upper_percentile = np.percentile(values, 40)
    for name, val in normalized_records.items():
        if lower_percentile <= val <= upper_percentile:
            ood_near.append(name)
            # ood_near.append((name, 0))

    true_labels = []
    for test_id in ood_near:
        true_labels.append(int(normalized_records[test_id] > median))

    res = get_score_prediction_per_model(ood_near, predictions)
    aucs = per_model_auc(true_labels, res)

    # mean ood near results
    pred_binary_level = []
    for test_id in ood_near:
        pred = predictions[test_id]
        pred = np.asarray(pred)
        pred = np.where(pred > 0.5, 1, 0)
        models_predictions = np.mean(pred, axis=1)
        models_predictions = np.where(models_predictions > 0.5, 1, 0)
        final_prediction = np.mean(models_predictions)
        # final_prediction = int(np.mean(models_predictions) > 0.5)
        pred_binary_level.append(final_prediction)
    auc = roc_auc_score(true_labels, pred_binary_level)

    indices = [0, 1, 4]
    # indices = [0, 2, 3]
    pred_binary_level = []
    for test_id in ood_near:
        pred = predictions[test_id]
        pred = np.asarray(pred)[indices, :]
        pred = np.where(pred > 0.5, 1, 0)
        models_predictions = np.mean(pred, axis=1)
        # models_predictions = np.where(models_predictions > 0.5, 1, 0)
        final_prediction = np.mean(models_predictions)
        pred_binary_level.append(final_prediction)
    auc = roc_auc_score(true_labels, pred_binary_level)

    # ood 40-60
    ood_middle = []
    values = list(normalized_records.values())
    lower_percentile = np.percentile(values, 40)
    upper_percentile = np.percentile(values, 60)
    for name, val in normalized_records.items():
        if lower_percentile <= val <= upper_percentile:
            ood_middle.append(name)

    true_labels = []
    for test_id in ood_middle:
        true_labels.append(int(normalized_records[test_id] > median))

    res = get_score_prediction_per_model(ood_middle, predictions)
    aucs = per_model_auc(true_labels, res)

    # mean ood near results
    pred_binary_level = []
    for test_id in ood_middle:
        pred = predictions[test_id]
        pred = np.asarray(pred)
        pred = np.where(pred > 0.5, 1, 0)
        models_predictions = np.mean(pred, axis=1)
        models_predictions = np.where(models_predictions > 0.5, 1, 0)
        final_prediction = np.mean(models_predictions)
        # final_prediction = int(np.mean(models_predictions) > 0.5)
        pred_binary_level.append(final_prediction)
    auc = roc_auc_score(true_labels, pred_binary_level)

    indices = [0, 1, 4]
    # indices = [0, 2, 3]
    pred_binary_level = []
    for test_id in ood_middle:
        pred = predictions[test_id]
        pred = np.asarray(pred)[indices, :]
        pred = np.where(pred > 0.5, 1, 0)
        models_predictions = np.mean(pred, axis=1)
        # models_predictions = np.where(models_predictions > 0.5, 1, 0)
        final_prediction = np.mean(models_predictions)
        pred_binary_level.append(final_prediction)
    auc = roc_auc_score(true_labels, pred_binary_level)



def main():
    parser = init_argparse()
    args = parser.parse_args()

    if args.train:
        seed_everything(Configuration.SEED)
        for gene in Configuration.GENES:
            prepare_train_env(gene)
            train(args, gene)


if __name__ == '__main__':
    main()
