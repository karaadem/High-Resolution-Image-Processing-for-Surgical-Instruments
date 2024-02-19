from transforms import get_augmentations, get_preprocess
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import csv
import os


class OP(Dataset):
    def __init__(self, train, path, image_size):
        super(OP, self).__init__()

        path_default = os.path.join(path, 'default.csv')

        self.data = {}
        with open(path_default, 'r') as f:
            reader = csv.reader(f)

            for idx, line in enumerate(reader):
                if line:
                    self.data[idx] = [line[0], int(line[1])]

        transform_list = []
        if train:
            transform_list += get_augmentations()

        transform_list += get_preprocess(image_size)

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)
