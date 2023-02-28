import os

import numpy as np
import skimage.transform
import torch
from skimage import io
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from models.pytorch.morpohlogical_extractor import MorphologicalFeatureExtractor
from models.pytorch.texture_extractor import TextureFeaturesExtractor


class TilesDataset(Dataset):
    """
    This class represent a dataset for a given list of slide ids.
    It is agnostic to the ids context, that is, it is unaware whether
     it is a train or val or test set.
    The ids should be determined by a different function
    """

    def __init__(self, tiles_directory, transform, gray2rgb_transform, ids):
        self.root_dir = tiles_directory
        self.transform = transform
        self.gray2rgb_transform = gray2rgb_transform
        self._files = self._load_files(ids)
        # a = "/Users/shahar.mor/git/H-E-Proteomics/data/images/zoom_20_size_512/PD31107a.ndpi_65_69.jpeg"
        # self._files = [a * 160]
        self._ids = ids
        self.morphological_feature = MorphologicalFeatureExtractor()
        self.texture_features = TextureFeaturesExtractor()

    def _load_files(self, ids):
        files = os.listdir(self.root_dir)
        filtered_files = list(filter(lambda x: any(x.startswith(str(prefix)) for prefix in ids.keys()), files))
        return filtered_files

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        # img_path = "/Users/shahar.mor/git/H-E-Proteomics/data/images/zoom_20_size_512/PD31107a.ndpi_65_69.jpeg"
        img_path = os.path.join(self.root_dir, self._files[index])
        img = io.imread(img_path)
        img = skimage.transform.resize(img, (224, 224), preserve_range=True).astype('uint8')
        morph_features = self.morphological_feature.extract(img)
        morph_features = self.gray2rgb_transform(torch.from_numpy(morph_features.astype(np.float32)))
        # textures_features = torch.from_numpy(self.texture_features.extract(img))
        # if textures_features.shape != torch.Size([15, ]):
        #     if textures_features.shape[0] > 15:
        #         print("Fuck {} shape: {}".format(img_path, textures_features.shape))
        #     else:
        #         pad_length = 15 - textures_features.shape[0]
        #         textures_features = torch.cat([textures_features, torch.zeros(pad_length)], dim=0)

        img = Image.open(img_path)
        img = self.transform(img)

        tile_slide_name = os.path.basename(self._files[index])  # every tile gets the label of the entire slide
        slide_id = tile_slide_name[:tile_slide_name.find(".")]
        tile_true_label = self._ids[slide_id]

        return img, morph_features, tile_true_label
        # return img, morph_features, textures_features, tile_true_label
