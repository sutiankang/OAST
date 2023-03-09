import os
import random
import numpy as np
import os.path as osp
from PIL import Image
from PIL import ImageEnhance

from torchvision import transforms
from torch.utils.data import Dataset


class UVOSDataset(Dataset):
    """
    image suffix: jpg
    flow suffix: jpg
    mask suffix: png
    """
    def __init__(self, data_dir, size: list, mean, std, mode, datasets, stride=None):

        self.labeled_support_dataset = ["YouTubeVOS-2018", "DAVIS-2016"]
        self.unlabeled_support_dataset = ["Youtube-objects"]
        self.test_support_dataset = ['DAVIS-2016', 'FBMS']

        self.datasets = datasets
        self.data_dir = data_dir
        if mode == 'labeled':
            self.stride = stride
        self.mode = mode
        self.size = size

        if self.mode in ['labeled', 'test']:
            self.images, self.flows, self.masks = [], [], []
        elif self.mode == 'unlabeled':
            self.images, self.flows = [], []
        else:
            raise ValueError

        # load dataset
        self.split_dataset(mode=mode)

        self.image_transform = self.get_image_transform(size=size, mean=mean, std=std)
        self.flow_transform = self.get_flow_mask_transform(size=size)
        if mode in ['labeled', 'test']:
            self.mask_transform = self.get_flow_mask_transform(size=size)
            assert len(self.images) == len(self.flows) == len(self.masks)
        else:
            assert len(self.images) == len(self.flows)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.mode in ['labeled', 'test']:
            image, flow, mask = self.images[idx], self.flows[idx], self.masks[idx]
            image = Image.open(image).convert("RGB")
            flow = Image.open(flow).convert("RGB")
            mask = Image.open(mask).convert("P")
            size = image.size
            image, flow, mask = self.data_augmentation(image, flow, mask)
            targets = {
                "image": image,
                "flow": flow,
                "mask": mask,
                "path": self.images[idx],
                "size": size
            }
            if self.mode == 'test':
                ori_image = Image.open(self.images[idx]).convert("RGB")
                for i in range(3):
                    aug_image = self.color_enhance(ori_image)
                    aug_image = self.image_transform(aug_image)
                    targets[f'aug_image{i+1}'] = aug_image
        else:
            image, flow = self.images[idx], self.flows[idx]
            image = Image.open(image).convert("RGB")
            flow = Image.open(flow).convert("RGB")
            size = image.size
            image, flow = self.data_augmentation(image, flow)
            targets = {
                "image": image,
                "flow": flow,
                "path": self.images[idx],
                "size": size
            }

        return targets

    def data_augmentation(self, image, flow, mask=None):

        if self.mode == 'labeled':
            image, flow, mask = self.random_crop(image, flow, mask, border=60)
            image, flow, mask = self.cv_random_flip(image, flow, mask)
            image, flow, mask = self.random_rotation(image, flow, mask)
            image = self.color_enhance(image)
            image = self.image_transform(image)
            flow = self.flow_transform(flow)
            mask = self.mask_transform(mask)
            return image, flow, mask
        elif self.mode == 'test':
            image = self.image_transform(image)
            flow = self.flow_transform(flow)
            mask = self.mask_transform(mask)
            return image, flow, mask
        elif self.mode == 'unlabeled':
            image, flow = self.random_crop(image, flow, border=60)
            image, flow = self.cv_random_flip(image, flow)
            image, flow = self.random_rotation(image, flow)
            image = self.color_enhance(image)
            image = self.image_transform(image)
            flow = self.flow_transform(flow)
            return image, flow

        else:
            raise ValueError

    def random_crop(self, image, flow, mask=None, border=30):
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_width = np.random.randint(image_width - border, image_width)
        crop_win_height = np.random.randint(image_height - border, image_height)
        random_region = (
            (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1,
            (image_width + crop_win_width) >> 1,
            (image_height + crop_win_height) >> 1)
        if mask is not None:
            return image.crop(random_region), flow.crop(random_region), mask.crop(random_region)
        else:
            return image.crop(random_region), flow.crop(random_region)

    def random_rotation(self, image, flow, mask=None):
        mode = Image.Resampling.BICUBIC
        if mask is not None:
            if random.random() > 0.5:
                random_angle = np.random.randint(-10, 10)
                image = image.rotate(random_angle, mode)
                flow = flow.rotate(random_angle, mode)
                mask = mask.rotate(random_angle, mode)
            return image, flow, mask
        else:
            if random.random() > 0.5:
                random_angle = np.random.randint(-10, 10)
                image = image.rotate(random_angle, mode)
                flow = flow.rotate(random_angle, mode)
            return image, flow

    def cv_random_flip(self, image, flow, mask=None):
        flip_flag = random.randint(0, 1)
        if mask is not None:
            if flip_flag == 1:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                flow = flow.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return image, flow, mask
        else:
            if flip_flag == 1:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                flow = flow.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return image, flow

    def color_enhance(self, image):
        bright_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Brightness(image).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        image = ImageEnhance.Color(image).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
        return image

    def split_dataset(self, mode):
        if mode == 'labeled':
            support_dataset = self.labeled_support_dataset
        elif mode == 'unlabeled':
            support_dataset = self.unlabeled_support_dataset
        elif mode == 'test':
            support_dataset = self.test_support_dataset
        else:
            raise ValueError

        for dataset in self.datasets:
            if dataset in support_dataset:
                self.load_dataset(dataset_name=dataset, mode=mode)
            else:
                raise ValueError(f"Not support this dataset: {dataset}")

    def load_dataset(self, dataset_name, mode):

        if mode in ['labeled']:
            data_dir = osp.join(self.data_dir, dataset_name, "train")
        elif mode in ['unlabeled', 'test']:
            data_dir = osp.join(self.data_dir, dataset_name, "val")
        else:
            raise ValueError

        assert os.listdir(osp.join(data_dir, "images")) == os.listdir(osp.join(data_dir, "flows")), \
            "video number or video name are different between images, flows."

        videos = os.listdir(osp.join(data_dir, "images"))
        for video in videos:
            # image
            image_dir = osp.join(data_dir, "images", video)
            images_frames = sorted(os.listdir(image_dir))
            # flow
            flow_dir = osp.join(data_dir, "flows", video)
            flow_frames = sorted(os.listdir(flow_dir))
            # mask
            frames = flow_frames
            if mode in ['labeled', 'test']:
                mask_dir = osp.join(data_dir, "labels", video)
                mask_frames = sorted(os.listdir(mask_dir))

                if len(mask_frames) < len(flow_frames):
                    frames = mask_frames

            if mode in ['labeled']:
                if self.stride > 1 and dataset_name == "YouTubeVOS-2018":
                    frames = frames[::self.stride]

            for frame in frames:
                assert frame[:-4] + ".jpg" in images_frames
                assert frame[:-4] + ".jpg" in flow_frames
                if mode  in ['labeled', 'test']:
                    assert frame[:-4] + ".png" in mask_frames
                    self.masks.append(osp.join(mask_dir, frame[:-4]) + ".png")
                self.images.append(osp.join(image_dir, frame[:-4] + ".jpg"))
                self.flows.append(osp.join(flow_dir, frame[:-4] + ".jpg"))

    def get_image_transform(self, size, mean, std):
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return transform

    def get_flow_mask_transform(self, size):
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        return transform