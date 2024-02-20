from torchvision import transforms
from PIL import ImageOps
import numpy as np
import random


class Rotate():
    def __init__(self):
        self.rotation = transforms.functional.rotate
        self.range = [-180, 180]

    def __call__(self, img):
        angle = random.uniform(self.range[0], self.range[1])
        return self.rotation(img, angle)


class ColorJitter():
    def __init__(self):
        self.ColorJitter = transforms.ColorJitter(brightness=0.2,
                                                  contrast=0.2,
                                                  saturation=0.2,
                                                  hue=0.05)

    def __call__(self, img):
        return self.ColorJitter(img)


class Flip():
    def __init__(self):
        self.flip = ImageOps.flip
        self.mirror = ImageOps.mirror

    def __call__(self, img):
        hflip = bool(random.choice([True, False]))
        vflip = bool(random.choice([True, False]))

        if hflip:
            img = self.flip(img)
        if vflip:
            img = self.mirror(img)
        return img


class PadAndResize():
    def __init__(self, size):
        self.size = size
        self.pad = transforms.Pad((0, 256, 0, 256))

    def __call__(self, img):
        if img.size != self.size:
            img = self.pad(img)
            img = img.resize(self.size)
        return img


class ToTensor():
    def __init__(self):
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        return self.toTensor(img)


class Normalize():
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, img):
        return self.normalize(img)


class Pad():
    def __init__(self):
        self.pad = transforms.Pad((0, 256, 0, 256))

    def __call__(self, img):
        return self.pad(img)


class Resize():
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img.resize(self.size)


def get_augmentations():
    return [Rotate(),
            ColorJitter(),
            Flip()]


def get_preprocess(size):
    return [PadAndResize(size=size),
            ToTensor(),
            Normalize(std=np.array([0.485, 0.456, 0.406]),
                      mean=np.array([0.229, 0.224, 0.225]))]
