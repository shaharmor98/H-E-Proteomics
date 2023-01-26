import os

from PIL import Image
from torch.utils.data import Dataset


class TilesDataset(Dataset):
    """
    This class represent a dataset for a given list of slide ids.
    It is agnostic to the ids context, that is, it is unaware whether
     it is a train or val or test set.
    The ids should be determined by a different function
    """

    def __init__(self, tiles_directory, transform, ids, caller=None):
        self.root_dir = tiles_directory
        self.transform = transform
        self._files = self._load_files(ids)
        if caller is None:
            print("No specific caller was defined")
        else:
            print("Initiating dataloader for: ", caller)
        print("Loading: {} files".format(len(self._files)))

    def _load_files(self, ids):
        files = os.listdir(self.root_dir)
        filtered_files = list(filter(lambda x: any(x.startswith(str(prefix[0])) for prefix in ids), files))
        return filtered_files

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self._files[index])

        # TODO- remove when running on actual env!!!
        if os.path.basename(img_path) == ".DS_Store":
            index += 1
            img_path = os.path.join(self.root_dir, self._files[index])

        img = Image.open(img_path)
        img = self.transform(img)

        tile_slide_name = os.path.basename(self._files[index])  # every tile gets the label of the entire slide
        tile_true_label = self.tiles_labeler.attach_label_to_tile(tile_slide_name)

        return img, tile_true_label
