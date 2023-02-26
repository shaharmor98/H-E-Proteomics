import torch
from efficientnet_pytorch import EfficientNet
from sklearn.svm import SVR


class ProteinQuantPredictor(torch.nn.Module):
    def __init__(self):
        super(ProteinQuantPredictor, self).__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.freeze_architecture()
        self.svr = SVR(kernel='rbf')

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
        x = torch.concatenate(image_features, features)
        y_pred = torch.from_numpy(self.svr.predict(x).float())
        return y_pred
