import pathlib

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, img_dir, crop=None):

        if isinstance(img_dir, str):
            img_dir = pathlib.Path(img_dir)

        self.img_dir = img_dir

        if crop is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(crop)
            ])

        self.img_path_list = sorted(list(self.img_dir.iterdir()))

        image = Image.open(self.img_path_list[0]).convert("RGB")

        tensor_image = self.transform(image)

        self._height = tensor_image.shape[-2]
        self._width = tensor_image.shape[-1]

    def __len__(self):
        return len(self.img_path_list)

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def __getitem__(self, idx):
        # print(self.img_path_list[idx])
        img = Image.open(self.img_path_list[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if img.shape[-2] < img.shape[-1]:
            img = img.permute(0, 2, 1)
        return img


class LSDIRDataset(Dataset):
    def __init__(self, img_dir, crop=None):

        if isinstance(img_dir, str):
            img_dir = pathlib.Path(img_dir)

        self.img_dir = img_dir

        if crop is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(crop)
            ])

        self.img_path_list = []
        for sub_folders in sorted(self.img_dir.iterdir()):
            for img_path in sorted(sub_folders.iterdir()):
                self.img_path_list.append(img_path)

        # self.img_path_list = sorted(list(self.img_dir.iterdir()))

        image = Image.open(self.img_path_list[0]).convert("RGB")

        tensor_image = self.transform(image)

        self._height = tensor_image.shape[-2]
        self._width = tensor_image.shape[-1]

    def __len__(self):
        return len(self.img_path_list)

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def __getitem__(self, idx):
        img = Image.open(self.img_path_list[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if img.shape[-2] < img.shape[-1]:
            img = img.permute(0, 2, 1)
        return img