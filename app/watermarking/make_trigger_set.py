from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import torch
import glob
import random
import torchvision

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

    dataset_path = str(Path.cwd()) + '/datasets/abstract'

    return TriggerDataset(dataset_path, trigger_set_size, wm_classes, transform, extensions=['.jpg'], to_rgb=True)

def get_ood_torchvision(trigger_set_size, wm_classes, transform=None, dataset_name='mnist', **kwargs):

    dataset_path = str(Path.cwd()) + '/datasets/torchvision/' + dataset_name.lower()

    DATASETS = {
    'caltech101': torchvision.datasets.Caltech101,
    'caltech256': torchvision.datasets.Caltech256,
    'celeba': torchvision.datasets.CelebA,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100,
    'cityscapes': torchvision.datasets.Cityscapes,
    'coco': torchvision.datasets.CocoDetection,
    'emnist': torchvision.datasets.EMNIST,
    'fakedata': torchvision.datasets.FakeData,
    'fmnist': torchvision.datasets.FashionMNIST,
    'flickr': torchvision.datasets.Flickr8k,
    'inaturalist': torchvision.datasets.INaturalist,
    'kitti': torchvision.datasets.Kitti,
    'kmnist': torchvision.datasets.KMNIST,
    'lfw': torchvision.datasets.LFWPeople,
    'lsun': torchvision.datasets.LSUN,
    'mnist': torchvision.datasets.MNIST,
    'omniglot': torchvision.datasets.Omniglot,
    'phototour': torchvision.datasets.PhotoTour,
    'place365': torchvision.datasets.Places365,
    'qmnist': torchvision.datasets.QMNIST,
    'sbd': torchvision.datasets.SBDataset,
    'sbu': torchvision.datasets.SBU,
    'semeion': torchvision.datasets.SEMEION,
    'stl10': torchvision.datasets.STL10,
    'svhn': torchvision.datasets.SVHN,
    'usps': torchvision.datasets.USPS,
    'voc': torchvision.datasets.VOCDetection,
    'widerface': torchvision.datasets.WIDERFace
    }
    # Not included (video datasets):
    # HMDB51
    # ImageNet
    # Kinetics-400
    # UCF101

    try:
        dataset = DATASETS[dataset_name.lower()](dataset_path, transform=transform, download=True)
    except KeyError:
        raise ValueError('unknown dataset name')

    if trigger_set_size > len(dataset):
            raise ValueError('Trigger set size is too big: the dataset is smaller than the trigger set size')


    subset, _ = torch.utils.data.random_split(dataset, [trigger_set_size, len(dataset) - trigger_set_size], generator=torch.Generator().manual_seed(42))

    return TriggerSubset(subset, wm_classes)


def get_dataset(wm_type, trigger_set_size, num_classes, wm_classes=None, height=224, width=224, mean=None, std=None, **kwargs):

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
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=(height, width)),
            transforms.ToTensor()
            ])

    try:
        triggger_set =  WM_METHODS[wm_type](trigger_set_size, wm_classes, transform, **kwargs)
    except KeyError:
        raise ValueError('unknown wm method')

    return triggger_set

if __name__ == '__main__':

    dataset = get_dataset('ood_abstract', 100, 10, [5, 7, 3], dataset_name='cifar10')
    print(f'size of the trigger set {len(dataset)}')

    sample_id = random.randrange(len(dataset))
    tensor, label = dataset[sample_id]
    t = transforms.ToPILImage()
    image = t(tensor)
    image.show()
    print(image)
    print(label)