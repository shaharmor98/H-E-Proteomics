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
Method A: Spearman correlation for NFKB2: SpearmanrResult(correlation=-0.07878787878787878, pvalue=0.8287173946974606)
Method B: Spearman correlation for NFKB2: SpearmanrResult(correlation=-0.07878787878787878, pvalue=0.8287173946974606)
Method C: Spearman correlation for NFKB2: SpearmanrResult(correlation=-0.07878787878787878, pvalue=0.8287173946974606)
Method D: Spearman correlation for NFKB2: SpearmanrResult(correlation=-0.07878787878787878, pvalue=0.8287173946974606)
Method E: Spearman correlation for NFKB2: SpearmanrResult(correlation=-0.07878787878787878, pvalue=0.8287173946974606)


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

"""
1- scatter plot of spearman's result
2- two samples, plot H&E vs models' predictions

Gene: NFKB2, distance: 0.38429598669478443; high: 0.4768618062047525; low: 0.09256581950996805
Gene: EIF2B3, distance: 0.3587589973068776; high: 0.594127424650473; low: 0.23536842734359537
Gene: MAPK13, distance: 0.3397049900873089; high: 0.5268369190694716; low: 0.1871319289821627
Gene: ATP5MK, distance: 0.3396272912843563; high: 0.4990920225159281; low: 0.15946473123157176
Gene: HDAC1, distance: 0.3382314727911813; high: 0.49625867240839916; low: 0.15802719961721784




Gene: RPS9, distance: 0.3820729407982868; high: 0.6142004822946795; low: 0.2321275414963927
Gene: PSMB6, distance: 0.3818373626373626; high: 0.6088157509157509; low: 0.2269783882783883
Gene: PYM1, distance: 0.38157019093086575; high: 0.4877746249045673; low: 0.10620443397370152
Gene: RBM39, distance: 0.38148682766869774; high: 0.4581988950471467; low: 0.07671206737844899
Gene: SNX9, distance: 0.3806357208730402; high: 0.5589861666154319; low: 0.17835044574239167
Gene: COA6, distance: 0.37925867694452176; high: 0.5452453091818296; low: 0.16598663223730784
Gene: MYH9, distance: 0.3787506470577205; high: 0.43521271733061784; low: 0.056462070272897354
Gene: NHP2, distance: 0.37788876276958; high: 0.574125362593013; low: 0.19623659982343294
Gene: RPL13A, distance: 0.37696736947213694; high: 0.6082682446379156; low: 0.23130087516577869
Gene: TPP1, distance: 0.37690111635023177; high: 0.5631580969699207; low: 0.18625698061968893
Gene: TALDO1, distance: 0.3761823673608322; high: 0.5549790414422197; low: 0.17879667408138744
Gene: NCL, distance: 0.37572951667275567; high: 0.42853473009022264; low: 0.05280521341746696
Gene: PLOD1, distance: 0.3743773392465235; high: 0.5248180965477046; low: 0.15044075730118112
Gene: PSMB7, distance: 0.3703825615163549; high: 0.5803796926824604; low: 0.20999713116610547
Gene: CHMP3, distance: 0.3687061008247309; high: 0.5834915433211352; low: 0.2147854424964043
Gene: RPL3, distance: 0.3681154211433409; high: 0.6078702892140391; low: 0.23975486807069818
Gene: RPL18A, distance: 0.3664737148157226; high: 0.6612565658895729; low: 0.2947828510738503
Gene: UBQLN4, distance: 0.36522139983947544; high: 0.5464620251831641; low: 0.18124062534368865
Gene: GPX1, distance: 0.36516993486220595; high: 0.557845257495907; low: 0.192675322633701
Gene: YKT6, distance: 0.3647675173382296; high: 0.4864820719376446; low: 0.121714554599415
Gene: ATXN2L, distance: 0.3641918205376038; high: 0.471014580842974; low: 0.10682276030537019
Gene: ADSL, distance: 0.36368625897801093; high: 0.43239121659189744; low: 0.0687049576138865
Gene: ABI1, distance: 0.3625725297785474; high: 0.6054410111177252; low: 0.24286848133917777
Gene: DIDO1, distance: 0.36038666259509866; high: 0.5347559469975351; low: 0.17436928440243651
Gene: HTT, distance: 0.36024033682673495; high: 0.44269826431624615; low: 0.08245792748951117
Gene: SNU13, distance: 0.36012664007382683; high: 0.589248485486562; low: 0.22912184541273511
Gene: HSPE1, distance: 0.3599855960651777; high: 0.5331033102075212; low: 0.1731177141423435
Gene: AP1S1, distance: 0.3589810444816526; high: 0.5115825456970051; low: 0.15260150121535251
Gene: TMSB10, distance: 0.35848148046490336; high: 0.5390937957294699; low: 0.18061231526456656
Gene: BLOC1S2, distance: 0.35823901561321003; high: 0.59920482906727; low: 0.24096581345405998
Gene: CCDC47, distance: 0.35764271085369503; high: 0.6617959733993918; low: 0.30415326254569675
Gene: MTX1, distance: 0.3572574245288497; high: 0.5390533766041047; low: 0.18179595207525498
Gene: RPL7L1, distance: 0.3572213527490473; high: 0.502696878969334; low: 0.1454755262202867
Gene: RAC1, distance: 0.35673134837947856; high: 0.5363669877121003; low: 0.17963563933262178
Gene: PRPF6, distance: 0.35590803055245646; high: 0.5818947828325779; low: 0.2259867522801214
Gene: CLIC2, distance: 0.3554639105749755; high: 0.5167098827687148; low: 0.16124597219373926
Gene: RELA, distance: 0.35511649123131994; high: 0.5377808084690584; low: 0.18266431723773846
Gene: NSFL1C, distance: 0.35440447687309523; high: 0.5423848462959908; low: 0.1879803694228956
Gene: RPS7, distance: 0.35354249067741983; high: 0.5298496717123204; low: 0.17630718103490065
Gene: SF3B5, distance: 0.35315535562211664; high: 0.5818257807777779; low: 0.22867042515566124
Gene: ADRM1, distance: 0.3524787603285594; high: 0.5824971201241775; low: 0.2300183597956181
Gene: LASP1, distance: 0.3524767325899748; high: 0.5869750775405437; low: 0.2344983449505689
Gene: CANX, distance: 0.3524289853258207; high: 0.5199971023909761; low: 0.16756811706515537
Gene: BOLA2B, distance: 0.3522214744106244; high: 0.5384580019527362; low: 0.18623652754211178
Gene: EPPK1, distance: 0.3519822305781131; high: 0.6053273553488097; low: 0.25334512477069654
Gene: GNG12, distance: 0.35031052934828627; high: 0.5791542651229862; low: 0.2288437357746999
Gene: TOMM34, distance: 0.3502967289126707; high: 0.4931482266958139; low: 0.14285149778314318
Gene: UBL4A, distance: 0.35020008572529715; high: 0.5336925631503685; low: 0.18349247742507138
Gene: SEC61G, distance: 0.35010278287394175; high: 0.48677217568135095; low: 0.1366693928074092
Gene: ATP5PD, distance: 0.34958382119660464; high: 0.38105885433438863; low: 0.031475033137784006
Gene: TERF2IP, distance: 0.34881989615769793; high: 0.5633961012992249; low: 0.21457620514152698
Gene: CIAO2B, distance: 0.34876368132067914; high: 0.5193956583396966; low: 0.17063197701901747
Gene: VPS29, distance: 0.34849843842558553; high: 0.5932537504148173; low: 0.24475531198923178
Gene: GSK3B, distance: 0.34723167216350276; high: 0.6698613239151323; low: 0.32262965175162955
Gene: GTF2I, distance: 0.34718374160068427; high: 0.40909458235546; low: 0.0619108407547757
Gene: PSME2, distance: 0.3469960076729468; high: 0.48349534690299023; low: 0.13649933923004345
Gene: PTMA, distance: 0.34646268118401824; high: 0.5215273146515417; low: 0.17506463346752343
Gene: IMP4, distance: 0.34635791159217233; high: 0.5191796640440961; low: 0.1728217524519238
Gene: CTPS2, distance: 0.3458809740851673; high: 0.5270050787863003; low: 0.181124104701133
Gene: SUN2, distance: 0.34526117843710824; high: 0.5106094163532525; low: 0.1653482379161443
Gene: ARHGDIA, distance: 0.3450684271858939; high: 0.5453250379718908; low: 0.2002566107859969
Gene: MRPL48, distance: 0.34470696073479623; high: 0.5308702945046174; low: 0.18616333376982122
Gene: DCPS, distance: 0.343719335938064; high: 0.5152547120502002; low: 0.17153537611213626
Gene: ALDH16A1, distance: 0.3436259373621525; high: 0.5901863427544728; low: 0.2465604053923203
Gene: CAP1, distance: 0.34262338584153584; high: 0.6523429722726671; low: 0.3097195864311313
Gene: ACTR1A, distance: 0.34165593149231277; high: 0.517039230686969; low: 0.17538329919465626
Gene: TOR1AIP1, distance: 0.3415430717760308; high: 0.5554300335926329; low: 0.21388696181660205
Gene: RPL21, distance: 0.3406683521130714; high: 0.5995327847676842; low: 0.25886443265461284
Gene: WDR33, distance: 0.34051240058635207; high: 0.4974613323603222; low: 0.15694893177397012
Gene: PSMD10, distance: 0.3404990364389807; high: 0.49048052095272876; low: 0.14998148451374807
Gene: RPS18, distance: 0.3404960550429174; high: 0.6053106814054348; low: 0.26481462636251735
Gene: MYO1D, distance: 0.34044959223729476; high: 0.5183896131687793; low: 0.1779400209314845
Gene: LSM3, distance: 0.34040701837824977; high: 0.6009890368240568; low: 0.26058201844580703
Gene: CYB5R3, distance: 0.33987193320691017; high: 0.5993464788500223; low: 0.2594745456431122
Gene: MTPN, distance: 0.3390018627069138; high: 0.4895029043430433; low: 0.1505010416361295
Gene: SRSF3, distance: 0.33877344734575643; high: 0.3923224570628718; low: 0.05354900971711533
Gene: NOP58, distance: 0.3385335664005291; high: 0.5476744460717153; low: 0.20914087967118625
Gene: HDAC1, distance: 0.3382314727911813; high: 0.49625867240839916; low: 0.15802719961721784
Gene: ADA, distance: 0.3379979422519515; high: 0.4727952597693155; low: 0.134797317517364
Gene: RPS25, distance: 0.33737260746822784; high: 0.5531767677771757; low: 0.21580416030894786
Gene: BPNT1, distance: 0.3373213206393363; high: 0.5891062513084916; low: 0.2517849306691553
Gene: MYL6, distance: 0.3372324596419951; high: 0.5188767882518672; low: 0.1816443286098721
Gene: XPO1, distance: 0.337204731255524; high: 0.5890142098364352; low: 0.25180947858091113
Gene: ATP6V1G1, distance: 0.3370449538772156; high: 0.5227461414486545; low: 0.18570118757143889
Gene: GNPDA2, distance: 0.3367489711934155; high: 0.5519645306682343; low: 0.2152155594748187
Gene: RABL3, distance: 0.336338918017251; high: 0.46629750480585874; low: 0.12995858678860772
Gene: PUM3, distance: 0.33567527343695147; high: 0.4622119259228058; low: 0.1265366524858543
Gene: IFI35, distance: 0.33566093152274407; high: 0.4993663542252116; low: 0.16370542270246752
Gene: STX7, distance: 0.3355543013752514; high: 0.5449942172311546; low: 0.2094399158559032
Gene: TBCE, distance: 0.3352625793887815; high: 0.5235973323248853; low: 0.18833475293610374
Gene: RPL38, distance: 0.33493959445592725; high: 0.5166527083702618; low: 0.18171311391433453
Gene: RPF2, distance: 0.33490255481137154; high: 0.4089698085747805; low: 0.07406725376340892
Gene: PURA, distance: 0.33460317460317457; high: 0.48842174774378166; low: 0.15381857314060707
Gene: NACA, distance: 0.33454347551221225; high: 0.5792370427046387; low: 0.24469356719242646
Gene: THUMPD2, distance: 0.334260533276286; high: 0.43208087814531787; low: 0.09782034486903186
Gene: CYP27A1, distance: 0.3341692512706817; high: 0.4464093658973784; low: 0.1122401146266967
Gene: TKT, distance: 0.3331271928893378; high: 0.5300176476489439; low: 0.19689045475960607
Gene: CALD1, distance: 0.33310335124517343; high: 0.4415218798957882; low: 0.10841852865061481
Gene: NMD3, distance: 0.3330365854275041; high: 0.4453554150173651; low: 0.11231882958986102
Gene: MCTS1, distance: 0.3323218559176965; high: 0.4691092582972092; low: 0.1367874023795127
Gene: FNDC3A, distance: 0.3314074192068026; high: 0.5147750761377041; low: 0.1833676569309015
Gene: SELENOF, distance: 0.33122120746787354; high: 0.4923444417504063; low: 0.16112323428253275
Gene: MCFD2, distance: 0.33072434871051226; high: 0.47445011731770437; low: 0.1437257686071921
Gene: PDLIM1, distance: 0.3307085774010381; high: 0.47870121020398015; low: 0.14799263280294203
Gene: PCNA, distance: 0.33052004487699405; high: 0.4867070349754293; low: 0.15618699009843526
Gene: PSMB4, distance: 0.33045882528769993; high: 0.6348506001873648; low: 0.3043917748996649
Gene: RABIF, distance: 0.33031150862763164; high: 0.5515481795266507; low: 0.22123667089901905
Gene: PSMB1, distance: 0.33016650589716673; high: 0.5640901087615617; low: 0.233923602864395
Gene: IRF2BP2, distance: 0.32986747354542073; high: 0.45486620575408226; low: 0.12499873220866153
Gene: DYNLRB1, distance: 0.32962604252207806; high: 0.459228076684882; low: 0.12960203416280394
Gene: UBA5, distance: 0.32918940492608495; high: 0.6275577825109361; low: 0.29836837758485113
Gene: SSB, distance: 0.3287700562621453; high: 0.525289879082384; low: 0.19651982282023872
Gene: DOCK2, distance: 0.3286164388547186; high: 0.4799191003291099; low: 0.1513026614743913
Gene: LSP1, distance: 0.32812296223021165; high: 0.43017126248653587; low: 0.10204830025632419
Gene: LMNA, distance: 0.32794586995794106; high: 0.49749733283280584; low: 0.16955146287486478
Gene: SAMHD1, distance: 0.3278671066950005; high: 0.5238310406250064; low: 0.19596393393000588
Gene: RPL13, distance: 0.3276889669062483; high: 0.6354628261132391; low: 0.3077738592069908
Gene: GFM1, distance: 0.3274956217162872; high: 0.4942848951434388; low: 0.1667892734271516
Gene: ARL6IP5, distance: 0.3269449666077855; high: 0.591231237542862; low: 0.2642862709350765
Gene: ANXA5, distance: 0.326919587256508; high: 0.6032535201130563; low: 0.27633393285654834
Gene: LEMD2, distance: 0.32665272448472715; high: 0.4909590589387144; low: 0.16430633445398724
Gene: SDF4, distance: 0.3265289793185241; high: 0.46038344006514553; low: 0.1338544607466214
Gene: CCDC50, distance: 0.3262961108100135; high: 0.45004081968675175; low: 0.12374470887673826
Gene: ARHGAP1, distance: 0.3262703348917379; high: 0.6372917890128663; low: 0.3110214541211284
Gene: RRBP1, distance: 0.3262164550919774; high: 0.5104716389773936; low: 0.18425518388541615
Gene: CTSS, distance: 0.32611178235309146; high: 0.4384414539286551; low: 0.11232967157556364
Gene: OSTF1, distance: 0.32597949380373803; high: 0.5637815363500586; low: 0.23780204254632056
Gene: EIF3J, distance: 0.32515706143825346; high: 0.6012683192500659; low: 0.2761112578118124
Gene: MBD2, distance: 0.32498397117021516; high: 0.5170226172315447; low: 0.19203864606132962
Gene: PDCD4, distance: 0.32475249908221526; high: 0.4868504643087124; low: 0.16209796522649716
Gene: GSN, distance: 0.32471099065862835; high: 0.6304548291455434; low: 0.30574383848691505
Gene: LMNB1, distance: 0.3244842007864909; high: 0.5516080499653019; low: 0.227123849178811
Gene: SSR1, distance: 0.3244479118132533; high: 0.5053532562566142; low: 0.1809053444433609
Gene: RPSA, distance: 0.32428753925867726; high: 0.6003451709408475; low: 0.27605763168217023
Gene: NUBP2, distance: 0.32420005539699165; high: 0.5798397100087912; low: 0.25563965461179955
Gene: PSMA6, distance: 0.3240750325608886; high: 0.6841932126393577; low: 0.36011818007846913
Gene: AARS2, distance: 0.32349000531279054; high: 0.532025335369903; low: 0.2085353300571125
Gene: GAR1, distance: 0.3234346715094791; high: 0.566981565881456; low: 0.2435468943719769
Gene: PRDX1, distance: 0.32340404865820715; high: 0.5448405484393927; low: 0.22143649978118554
Gene: FDXR, distance: 0.32328333021408606; high: 0.43787918679638504; low: 0.11459585658229895
Gene: PPIH, distance: 0.3232303053324141; high: 0.521950075735194; low: 0.19871977040277994
Gene: SRP72, distance: 0.32321058127043933; high: 0.557551075663053; low: 0.2343404943926136
Gene: NCBP2, distance: 0.3231650244663403; high: 0.4748310022533032; low: 0.15166597778696286
Gene: SCYL1, distance: 0.32298482600661094; high: 0.44538559641862524; low: 0.12240077041201429
Gene: RBM42, distance: 0.32296803703662585; high: 0.46366169662384954; low: 0.14069365958722369
Gene: TMEM68, distance: 0.32285852698393325; high: 0.4624212301072644; low: 0.13956270312333116
Gene: RPS6, distance: 0.32284665295482917; high: 0.6249853761365505; low: 0.3021387231817213
Gene: STXBP3, distance: 0.32240308139588714; high: 0.6016268066627779; low: 0.27922372526689077
Gene: SERPINB8, distance: 0.32230150114176614; high: 0.5285329240444727; low: 0.20623142290270655
Gene: GLIPR2, distance: 0.322211619774644; high: 0.5372068772427033; low: 0.2149952574680593
Gene: FLNC, distance: 0.3221709160708126; high: 0.4406309622968297; low: 0.11846004622601711
Gene: MORC2, distance: 0.322142137751441; high: 0.3981581754651938; low: 0.07601603771375277
Gene: MAPRE1, distance: 0.3219925027088506; high: 0.5643850628475797; low: 0.2423925601387291
Gene: TUBG1, distance: 0.3217334894731964; high: 0.5348997868522372; low: 0.21316629737904075
Gene: ATG5, distance: 0.3217295007499234; high: 0.537459076760976; low: 0.21572957601105264
Gene: KPNA4, distance: 0.3216597289055343; high: 0.45246454784736145; low: 0.13080481894182713
Gene: COX5B, distance: 0.32145248599215975; high: 0.5887636410249709; low: 0.26731115503281117
Gene: U2SURP, distance: 0.32131847210678477; high: 0.5287431181581171; low: 0.20742464605133235
Gene: SMAP2, distance: 0.32100429975429967; high: 0.45391277641277633; low: 0.13290847665847663
Gene: BLOC1S1, distance: 0.3209460925633156; high: 0.6531603583496077; low: 0.33221426578629215
Gene: LARS1, distance: 0.3207482898963713; high: 0.5579023608335146; low: 0.23715407093714333
Gene: RFC2, distance: 0.31960233116215286; high: 0.5181093589304079; low: 0.19850702776825505
Gene: PIN4, distance: 0.3194041825754247; high: 0.5315285712557806; low: 0.21212438868035588
Gene: RAP1A, distance: 0.31925203544396585; high: 0.46132049749674664; low: 0.1420684620527808
Gene: ARPC5, distance: 0.3190012818328661; high: 0.5879559639847783; low: 0.26895468215191226
Gene: HSPA6, distance: 0.31891309190189543; high: 0.5082271989979535; low: 0.1893141070960581
Gene: RAB11B, distance: 0.31868989252830615; high: 0.5534765323460202; low: 0.23478663981771408
Gene: HDHD3, distance: 0.3183572238789587; high: 0.5393599407233942; low: 0.22100271684443554
Gene: AP2B1, distance: 0.31833458141718; high: 0.600099540429494; low: 0.28176495901231396
Gene: CAPNS1, distance: 0.3178391846130622; high: 0.5979541990112914; low: 0.2801150143982292
Gene: NMT1, distance: 0.31723741602192884; high: 0.5732273576572645; low: 0.25598994163533567
Gene: SNF8, distance: 0.3162195665648141; high: 0.4422093668105651; low: 0.12598980024575102
Gene: DLST, distance: 0.3161796638849651; high: 0.4722451276737694; low: 0.15606546378880434
Gene: PABPC1, distance: 0.31596964318808524; high: 0.5919322426038742; low: 0.275962599415789
Gene: GAPDH, distance: 0.31564186749154904; high: 0.4583166123459823; low: 0.14267474485443327
Gene: THOC6, distance: 0.31540462124973234; high: 0.5446898442730845; low: 0.22928522302335216
Gene: ATP5F1A, distance: 0.315371333815199; high: 0.5099900288513065; low: 0.19461869503610751
Gene: YWHAZ, distance: 0.31522846983785235; high: 0.47128300019670355; low: 0.15605453035885117

"""
