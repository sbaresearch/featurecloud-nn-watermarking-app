from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import torch
import glob
import random
import torchvision

class SetChannels:
    def __init__(self, num_channels):
        assert num_channels in (1, 3)
        self.channels = num_channels

    def __call__(self, tensor):
        tensor_channels = tensor.size()[0]
        assert tensor_channels in (1, 3)
        if tensor_channels == self.channels:
            return tensor
        
        if tensor_channels == 3:
            assert self.channels == 1
            return transforms.functional.rgb_to_grayscale(tensor, self.channels)
            
        if tensor_channels == 1:
            assert self.channels == 3
            return torch.cat([tensor, tensor, tensor], dim=0)


class TriggerDataset(Dataset):
    
    def __init__(self, dataset_path, trigger_set_size, wm_classes, transform, extensions=['.jpg', '.jpeg', '.png'], to_rgb=False):
        self.img_paths = []
        self.img_labels = []
        self.transform = transform
        self.to_rgb = to_rgb

        img_paths_full = []
        for extension in extensions:
            for image_path in glob.glob(dataset_path+'/*'+extension):
                img_paths_full.append(image_path)

        if trigger_set_size > len(img_paths_full):
            raise ValueError(f'Trigger set size is too big: the dataset {len(img_paths_full)} is smaller than the trigger set size {trigger_set_size}')

        self.img_paths = random.sample(img_paths_full, trigger_set_size)

        for i in range(len(self.img_paths)):
            self.img_labels.append(wm_classes[i % len(wm_classes)])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        image = Image.open(self.img_paths[idx])
        if self.to_rgb:
            image = image.convert('RGB')
        label = self.img_labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


class TriggerSubset(Dataset):
    def __init__(self, subset, wm_classes):
        self.images = []
        self.img_labels = []

        for i in range(len(subset)):
            image, _ = subset[i]
            self.images.append(image)
            self.img_labels.append(wm_classes[i % len(wm_classes)])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return self.images[idx], self.img_labels[idx] 


def get_custom(trigger_set_size, wm_classes, transform, **kwargs):
    dataset_path = kwargs['dataset_path']
    extensions = kwargs['extensions']
    return TriggerDataset(dataset_path, trigger_set_size, wm_classes, transform, extensions)


def get_ood_abstract(trigger_set_size, wm_classes, transform, **kwargs):
    if trigger_set_size > 100:
        raise ValueError('for this wm method trigger set size should be less or equal 100')

    dataset_path = str(Path.cwd()) + 'mnt/input/datasets/abstract'

    return TriggerDataset(dataset_path, trigger_set_size, wm_classes, transform, extensions=['.jpg'], to_rgb=True)

def get_ood_torchvision(trigger_set_size, wm_classes, transform=None, dataset_name='mnist', **kwargs):

    dataset_path = str(Path.cwd()) + 'mnt/input/datasets/' + dataset_name.lower()

    DATASETS = {
    'caltech101': torchvision.datasets.Caltech101,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100,
    'fmnist': torchvision.datasets.FashionMNIST,
    'kmnist': torchvision.datasets.KMNIST,
    'lfw': torchvision.datasets.LFWPeople,
    'mnist': torchvision.datasets.MNIST,
    'omniglot': torchvision.datasets.Omniglot,
    'qmnist': torchvision.datasets.QMNIST,
    'semeion': torchvision.datasets.SEMEION,
    'svhn': torchvision.datasets.SVHN,
    'usps': torchvision.datasets.USPS
    }

    try:
        dataset = DATASETS[dataset_name.lower()](dataset_path, transform=transform, download=False)
    except KeyError:
        raise ValueError('unknown dataset name')

    if trigger_set_size > len(dataset):
            raise ValueError('Trigger set size is too big: the dataset is smaller than the trigger set size')


    subset, _ = torch.utils.data.random_split(dataset, [trigger_set_size, len(dataset) - trigger_set_size], generator=torch.Generator().manual_seed(42))

    return TriggerSubset(subset, wm_classes)


def get_dataset(wm_type, trigger_set_size, num_classes, num_channels, wm_classes=None, height=224, width=224, mean=None, std=None, **kwargs):

    WM_METHODS = {
        'custom': get_custom,
        'ood_abstract': get_ood_abstract,
        'ood_torchvision': get_ood_torchvision
    }

    if wm_classes:
        assert (len(wm_classes) <= num_classes and max(wm_classes) < num_classes)
    else:
        wm_classes = list(range(num_classes))

    if mean and std:
        transform = transforms.Compose([
            transforms.Resize(size=(height, width)),
            transforms.Normalize(mean, std),
            transforms.ToTensor(),
            SetChannels(num_channels)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=(height, width)),
            transforms.ToTensor(),
            SetChannels(num_channels)
            ])

    try:
        triggger_set =  WM_METHODS[wm_type](trigger_set_size, wm_classes, transform, **kwargs)
    except KeyError:
        raise ValueError('unknown wm method')

    return triggger_set
