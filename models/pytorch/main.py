import json
import multiprocessing
import random

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_parser.dia_to_metadata_parser import DiaToMetadata
from host_configuration import HostConfiguration
from models.pytorch.morpohlogical_extractor import MorphologicalFeatureExtractor
from models.pytorch.protein_quant_predictor import ProteinQuantPredictor
from models.pytorch.texture_extractor import TextureFeaturesExtractor
from models.pytorch.tiles.tiles_dataset import TilesDataset
from torchvision import transforms
import wandb
from tqdm import tqdm


def get_random_split(items, proportion):
    """
    Input expected to be a dict
    """
    keys = list(items.keys())
    random.shuffle(keys)
    split_idx = int(len(keys) * proportion)
    test_keys = keys[:split_idx]
    train_keys = keys[split_idx:]
    return {k: items[k] for k in test_keys}, {k: items[k] for k in train_keys}


def train(gene):
    wandb.init()
    torch.manual_seed(42)
    random.seed(42)

    random_image = np.random.rand(512, 512, 3)
    morph_features = MorphologicalFeatureExtractor().extract(random_image)
    textures_features = TextureFeaturesExtractor().extract(random_image)
    features = np.concatenate([morph_features, textures_features])

    model = ProteinQuantPredictor(features.shape[0])
    epochs = 2

    criterion = nn.MSELoss()
    optimizer = optim.Adagrad(model.parameters(), lr=0.001)

    device = "cuda"
    tiles_directory_path = HostConfiguration.TILES_DIRECTORY.format(zoom_level=HostConfiguration.ZOOM_LEVEL,
                                                                    patch_size=HostConfiguration.PATCH_SIZE)

    num_of_workers = int(multiprocessing.cpu_count())

    dia_metadata = DiaToMetadata(HostConfiguration.DIA_GENES_FILE_PATH, HostConfiguration.RNR_METADATA_FILE_PATH,
                                 tiles_directory_path)
    gene_slides_with_labels = dia_metadata.get_continuous_normalized_records(gene)

    transform_compose = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(size=(224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.], std=[255.])])
    test_proportion_size = 0.1
    val_proportion_size = 0.1

    train_set, test_set = get_random_split(gene_slides_with_labels, test_proportion_size)
    train_set, val_set = get_random_split(train_set, val_proportion_size)

    with open(HostConfiguration.TEST_IDS_FILE.format(gene=gene), 'w') as f:
        json.dump(test_set, f)
    print("Test: ", test_set)

    train_dataset = TilesDataset(tiles_directory_path, transform_compose, train_set)
    val_dataset = TilesDataset(tiles_directory_path, transform_compose, val_set)
    test_dataset = TilesDataset(tiles_directory_path, transform_compose, test_set)
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=num_of_workers,
                              persistent_workers=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=16, num_workers=num_of_workers,
                            persistent_workers=True, pin_memory=True, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=16, num_workers=num_of_workers,
                             persistent_workers=True, pin_memory=True, shuffle=True)

    wandb.watch(model, log_freq=10)
    model = model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = [inputs[0].to(device), inputs[1].to(device)]
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs).float()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                wandb.log({"epoch": epoch, "loss": running_loss / 2000})
                running_loss = 0.0


def main():
    train("STAT1")


if __name__ == '__main__':
    main()
