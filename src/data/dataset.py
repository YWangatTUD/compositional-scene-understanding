import os
import torch
import json
import random
import math
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import imageio
import os.path as osp
import pandas as pd


def compact(l):
    return list(filter(None, l))

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


class CLEVR2DPosBlender(Dataset):
    def __init__(
        self,
        resolution,
        data_root,
        split,
        ood=False,
        use_captions=False,
        random_crop=False,
        random_flip=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.use_captions = use_captions
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.data_root = data_root
        self.split = split
        self.clevr_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda X: 2 * X - 1.0),  # rescale between -1 and 1
            transforms.Resize((self.resolution,self.resolution),antialias=True),
        ]
    )


        if ood:
            self.n_objects_upper = 8
            self.n_objects_lower = 6
        else:
            self.n_objects_upper = 5
            self.n_objects_lower = 3

        if self.split == 'train':
            self.data_path = os.path.join(data_root, "images/train")
            self.json_path = os.path.join(self.data_root, f"CLEVR_train_scenes.json")
            self.max_num = 50000
        elif self.split == 'val':
            self.data_path = os.path.join(data_root, "images/val")
            self.json_path = os.path.join(self.data_root, f"CLEVR_val_scenes.json")
            self.max_num = 100


        assert os.path.exists(self.data_path), f"Path {self.data_path} does not exist"
        self.files, self.label, self.mask = self.get_files()
        print(f'number of samples: {len(self.files)}')

    def __getitem__(self, index):
        image_path = self.files[index]
        label = self.label[image_path]
        mask = self.mask[image_path]
        out_dict = {"y": label, "mask": mask}
        img = Image.open(image_path)
        img = img.convert("RGB")

        return self.clevr_transforms(img), out_dict

    def __len__(self):
        return len(self.files)

    def get_files(self):
        with open(self.json_path) as f:
            scene = json.load(f)
        paths = []
        label = {}
        mask = {}
        total_num_images = len(scene["scenes"])
        i = 0
        while len(paths) < self.max_num and i < total_num_images:
            num_objects_in_scene = len(scene["scenes"][i]["objects"])
            if num_objects_in_scene <= self.n_objects_upper and num_objects_in_scene >= self.n_objects_lower:
                image_path = os.path.join(self.data_path, scene["scenes"][i]["image_filename"])
                assert os.path.exists(image_path), f"{image_path} does not exist"
                paths.append(image_path)
                coords = []
                mask_obj = torch.zeros(1,self.n_objects_upper)
                for k in range(self.n_objects_upper):
                    if k < len(scene["scenes"][i]["objects"]):
                        scene["scenes"][i]["objects"][k]["pixel_coords"][0] = scene["scenes"][i]["objects"][k]["pixel_coords"][0]/480
                        scene["scenes"][i]["objects"][k]["pixel_coords"][1] = scene["scenes"][i]["objects"][k]["pixel_coords"][1]/320
                        coords.append(scene["scenes"][i]["objects"][k]["pixel_coords"][0:2])
                        mask_obj[:, k] = 1.
                    else:
                        coords.append([1, 1])
                label[image_path] = torch.tensor(coords)
                mask[image_path] = mask_obj
            i += 1
        return sorted(compact(paths)), label, mask


class CelebA(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.path = data_root + "/img_align_celeba/img_align_celeba"
        self.labels = pd.read_csv(data_root + "/list_attr_celeba.csv", sep="\s+", skiprows=1)
        self.labels = self.labels[:70000].reset_index()
        self.files, self.label = self.get_files()


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_path = self.files[index]
        im = imageio.imread(self.files[index])
        im = np.array(Image.fromarray(im).resize((128, 128)))
        im = im / 255.
        label = self.label[image_path]

        return im, label

    def get_files(self):
        paths = []
        label = {}
        i = 0
        while len(paths) < 70000 and i < self.labels.shape[0]:
            info = self.labels.iloc[i]
            fname = info.iloc[1].split(",")[0]
            if int(info.iloc[1].split(",")[21]) == -1: # -1 is female label
                image_path = osp.join(self.path, fname)
                assert os.path.exists(image_path), f"{image_path} does not exist"
                paths.append(image_path)

                label_blackhair = int(info.iloc[1].split(",")[9])
                if label_blackhair == -1:
                    label_blackhair = 0
                label_blackhair = np.eye(6)[label_blackhair]

                label_glasses = int(info.iloc[1].split(",")[16])
                if label_glasses == -1:
                    label_glasses = 0
                label_glasses = np.eye(6)[label_glasses + 2]

                label_smile = int(info.iloc[1].split(",")[32])
                if label_smile == -1:
                    label_smile = 0
                label_smile = np.eye(6)[label_smile + 4]


                label_cat = np.array([label_blackhair, label_glasses, label_smile])

                label[image_path] = label_cat
            i += 1
        return sorted(compact(paths)), label
