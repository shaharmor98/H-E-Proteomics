import torch
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

class ProteinQuantPredictor(torch.nn.Module):
    def __init__(self, features_size):
        super(ProteinQuantPredictor, self).__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.freeze_architecture()
        self.fc1 = torch.nn.Linear(1000 + features_size, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 1)
        # only architecture
        # model = EfficientNet.from_name('efficientnet-b0', num_classes=1000, in_channels=3)

        # num_ftrs = self.model._fc.in_features
        # self.model._fc = nn.Linear(num_ftrs, 1)
        # self.model._fc.bias.data.fill_(0.0)
        # self.model._fc.weight.data.normal_(0.0, 0.02)

    def freeze_architecture(self):
        for name, param in self.model.named_parameters():
            if not '_fc' in name:  # exclude the final layer
                param.requires_grad = False

    def forward(self, x):
        img, features = x
        image_features = self.model(img)
        x = torch.concatenate([image_features, features], dim=1)
        print("X, shape: ", x.shape)
        x = torch.reshape(x, (x.shape[1],)).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        pred = self.fc4(x)
        return pred
