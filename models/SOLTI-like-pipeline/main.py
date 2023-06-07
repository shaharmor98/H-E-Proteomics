import argparse
import multiprocessing
import os

from lightning_lite import seed_everything
from pytorch_lightning.loggers import WandbLogger

from configuration import Configuration
from data_parser.dia_to_metadata_parser import DiaToMetadata


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

    wandb_logger = WandbLogger(project="proteomics-project", log_model=True)
    num_of_workers = int(multiprocessing.cpu_count())

    dia_metadata = DiaToMetadata(Configuration.DIA_GENES_FILE_PATH, Configuration.RNR_METADATA_FILE_PATH,
                                 tiles_directory_path)
    gene_slides_with_labels = dia_metadata.get_gene_slides_with_labels(gene)


def main():
    parser = init_argparse()
    args = parser.parse_args()

    if args.train:
        seed_everything(Configuration.SEED)
        for gene in Configuration.GENES:
            prepare_train_env(gene)
            train(args, gene)

    elif args.inference:
        inference(args.gene)
    elif args.analysis:
        analysis(args.gene)


if __name__ == '__main__':
    main()
