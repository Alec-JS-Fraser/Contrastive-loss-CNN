from pathlib import Path
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from typing import Callable


BASE_IMAGE_PATH = Path("/home/irisdc01/training/object-tracking/feature-training-data")
random_seed = np.random.randint(0, 100000)
rng = np.random.default_rng(random_seed)
#  similar = 0 dissimialr = 1


def _get_pair(negative_pair_data: pd.DataFrame, index: int) -> tuple:
    farm_identifier, video_name, first_track_id, second_track_id = negative_pair_data.iloc[index]

    if first_track_id == second_track_id:
        image_path = BASE_IMAGE_PATH / f"{farm_identifier}/{video_name}/{first_track_id}"
        images = [img for img in image_path.glob("*")]

        choice1, choice2 = rng.integers(0, len(images), size=2)
        label = 0
        return images[choice1], images[choice2], label

    else:
        image_path1 = BASE_IMAGE_PATH / f"{farm_identifier}/{video_name}/{first_track_id}"
        image_path2 = BASE_IMAGE_PATH / f"{farm_identifier}/{video_name}/{second_track_id}"

        first_track_images = [img for img in image_path1.glob("*")]
        second_track_images = [img for img in image_path2.glob("*")]

        label = 1
        return (first_track_images[rng.integers(0, len(first_track_images))],
                second_track_images[rng.integers(0, len(second_track_images))], label)


class ImagePairDataset(Dataset):
    def _get_images(self, _get_pair: Callable, pair_data: pd.DataFrame, index: int) -> tuple:
        image_path1, image_path2, label = _get_pair(pair_data, index)

        image1 = np.array(Image.open(image_path1), dtype=np.float32) / 255.
        image2 = np.array(Image.open(image_path2), dtype=np.float32) / 255.

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label

    def __init__(self, pair_data: pd.DataFrame, transform=None):
        self.pair_data = pair_data
        self.transform = transform

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, index):
        image1, image2, label = self._get_images(_get_pair, self.pair_data, index)
        return image1, image2, label
