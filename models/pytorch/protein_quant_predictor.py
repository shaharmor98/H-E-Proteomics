import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torch import nn
from torchvision import transforms


class ProteinQuantPredictor(pl.LightningModule):
    def __init__(self, textures_features_size, device):
        super(ProteinQuantPredictor, self).__init__()

        self.image_features = EfficientNet.from_pretrained('efficientnet-b0')
        self.morphological_features = EfficientNet.from_pretrained('efficientnet-b0')
        self.freeze_architecture(self.image_features)
        self.freeze_architecture(self.morphological_features)
        self.fc1 = torch.nn.Linear(1000 + 1000 + textures_features_size, 256)
        self.fc2 = torch.nn.Linear(256, 1)
        self.learning_rate = 0.001
        self._device = device
        self.loss = nn.MSELoss()
        self.save_hyperparameters()

        # only architecture
        # model = EfficientNet.from_name('efficientnet-b0', num_classes=1000, in_channels=3)

        # num_ftrs = self.model._fc.in_features
        # self.model._fc = nn.Linear(num_ftrs, 1)
        # self.model._fc.bias.data.fill_(0.0)
        # self.model._fc.weight.data.normal_(0.0, 0.02)

    def freeze_architecture(self, model):
        for name, param in model.named_parameters():
            if not '_fc' in name:  # exclude the final layer
                param.requires_grad = False

    def forward(self, img, morph_features, textures_features):
        image_features = self.image_features(img)
        gray_to_rgb_transforms = transforms.Compose([
            transforms.ToPILImage(),  # convert tensor to PIL Image
            transforms.Grayscale(num_output_channels=3),  # convert grayscale to RGB
            transforms.ToTensor(),  # convert PIL Image to tensor
        ])

        tmp = morph_features[0]
        print("Tmp shape: ".format(tmp.shape))
        tmp = gray_to_rgb_transforms(tmp).to(self._device)
        print("Tmp shape: ".format(tmp.shape))
        morph_features = self.morphological_features(tmp)
        print("image features: ", image_features.shape)
        print("morph features: ", morph_features.shape)
        print("texture features: ", textures_features.shape)
        x = torch.concatenate([image_features, morph_features, textures_features], dim=1).float()
        x = F.relu(self.fc1(x))
        pred = self.fc2(x)
        return pred

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, label = batch
        y_hat = self(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        img, morph_features, textures_features, labels = batch

        original_labels = labels.reshape(-1, 1).float()
        if len(img) == 1:
            print("Found length 0")
            return

        y_hat = self(img, morph_features, textures_features)
        loss = self.loss(y_hat.float(), original_labels)

        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, original_labels = batch
        original_labels = original_labels.reshape(-1, 1).float()
        if len(x[0]) == 1:
            return
        y_hat = self(x)
        val_loss = self.loss(y_hat.float(), original_labels)

        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
        return {"loss": val_loss}

    def validation_epoch_end(self, outputs):
        losses = []
        for output in outputs:
            losses.append(output["loss"].cpu())

        self.log("val_epoch_loss", np.asarray(losses).mean(), sync_dist=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, original_labels = batch
        original_labels = original_labels.reshape(-1, 1).float()
        y_hat = self(x)
        test_loss = self.loss(y_hat.float(), original_labels)

        self.log("test_loss", test_loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

