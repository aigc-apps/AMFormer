import torchvision
import random
import numbers
import math
import torch
from torchvision.transforms import functional as F
import PIL
import numpy as np
from PIL import Image, ImageOps
from torch.autograd import Variable


class GroupResizeResize(torch.nn.Module):
    

    def __init__(self, size, interpolation, max_size=None, antialias=None):
        super().__init__()
        self.size = size
        self.max_size = max_size
        interpolation = torchvision.transforms.InterpolationMode.BILINEAR
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img_group):

        return [F.resize(img, self.size, self.interpolation, self.max_size, self.antialias) for img in img_group]

    
class GroupCenterCrop(torch.nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img_group):
        return [F.center_crop(img, self.size) for img in img_group]

class GroupRandomResizedCrop(torch.nn.Module):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=torchvision.transforms.InterpolationMode.BILINEAR):
        super().__init__()
        self.size = size
        interpolation = torchvision.transforms.InterpolationMode.BILINEAR
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        _, height, width = F.get_dimensions(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
    
    # def __init__(self, size, scale=..., ratio=..., interpolation=2):
        # super(torchvision.transforms.RandomResizedCrop, self).__init__(size, scale, ratio, interpolation)
    
    def forward(self, img_group):

        i, j, h, w = self.get_params(img_group[0], self.scale, self.ratio)
        return [F.resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in img_group]
    

class GroupRandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img_group):

        if torch.rand(1) < self.p:
            return [F.hflip(img) for img in img_group]
        return img_group

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class GroupNormalize(torch.nn.Module):

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, img_group):
        # tmp = []
        # for i in range(len(img_group)):
        #     tmp.append(F.normalize(img_group[i], self.mean[i], self.std[i], self.inplace))
        # return tmp
        return [F.normalize(img_group[0], self.mean[0], self.std[0], self.inplace), *img_group[1:]]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"