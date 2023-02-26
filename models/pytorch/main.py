import multiprocessing

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models.pytorch.protein_quant_predictor import ProteinQuantPredictor
from models.pytorch.tiles.tiles_dataset import TilesDataset


def main():
    torch.manual_seed(42)

    model = ProteinQuantPredictor()
    epochs = 2

    criterion = nn.MSELoss()
    optimizer = optim.Adagrad(model.parameters(), lr=0.001)
    num_of_workers = int(multiprocessing.cpu_count())

    test_id = "ASD"
    tiles_directory = ""
    transform_compose = None
    tiles_labeler = None
    dataset = TilesDataset(tiles_directory, transform_compose, tiles_labeler, [test_id])
    train_loader = DataLoader(dataset, batch_size=16, num_workers=num_of_workers,
                              persistent_workers=True, pin_memory=True, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0


if __name__ == '__main__':
    main()
