from torch.utils.data import Dataset
from collections import Counter, defaultdict
from torchvision import transforms
from PIL import Image
import torch
import json
import pickle
import glob

DATA_NAMES = {
    'train': {
        'images': 'images/train2014'
    },
    'val': {
        'images': 'images/val2014'
    }
}

class VQAImageDataset(Dataset):
    def __init__(self, root_path, preprocess, data_type='train'):
        super(VQAImageDataset, self).__init__()

        self.imgs_pathes = glob.glob(f'{root_path}/{DATA_NAMES[data_type]["images"]}/*.jpg')
        self.clip_img_preprocess = preprocess

    def __getitem__(self, index):
        img_path = self.imgs_pathes[index]
        image = self.clip_img_preprocess(Image.open(img_path))
        image_id = int(img_path.split('_')[-1].split('.')[0])

        return image, image_id

    def __len__(self):
        return len(self.imgs_pathes)