import os
from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class GANDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        img_size: int | tuple[int, int] = (64, 64),
        img_types: str = ["jpg"],  # , "png"],
    ) -> None:
        super().__init__()

        self.file_path = file_path

        for img_type in img_types:
            self.file_names = [
                file for file in glob(os.path.join(file_path, f"*.{img_type}"))
            ]

        print(len(self.file_names))

        self.transforms = T.Compose(
            {
                T.Resize(img_size, antialias=True),
                T.ToTensor(),
                # T.ColorJitter(brightness=0.2, hue=0.1),
                # T.RandomPerspective(distortion_scale=0.6, p=0.5),
                # T.RandomRotation(degrees=(0, 10)),
                # T.RandomHorizontalFlip(p=0.5),
            }
        )

    def __len__(self):
        return len(self.file_names)

    def normalize_tensor(self, data):
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

    def __getitem__(self, index):
        # Read image usin PIL
        img_path = os.path.join(self.file_path, self.file_names[index])
        img = Image.open(img_path)
        img = img.convert("RGB")
        return self.normalize_tensor(self.transforms(img))
