import csv
import torch

from pathlib import Path
from torchvision import transforms

def save_model(model, save_dir):
    path = save_dir + '/wm_model.pth'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def save_trigger_set(trigger_set, save_dir):
    images_folder = save_dir + '/images/'
    tensors_folder = save_dir + '/tensors/'

    Path(images_folder).mkdir(parents=True, exist_ok=True)
    Path(tensors_folder).mkdir(parents=True, exist_ok=True)

    transform = transforms.ToPILImage()
    labels = {}

    for idx, (data, label) in enumerate(trigger_set):
        torch.save(data, tensors_folder+str(idx))

        image = transform(data)
        image.save(images_folder+str(idx)+'.png')

        labels[idx] = label

    with open(save_dir+'/labels.csv', 'w') as f:
        writer = csv.writer(f)
        for idx, label in labels.items():
            writer.writerow([idx, label])


if __name__ == '__main__':
    from make_trigger_set import get_dataset
    trigger_set = get_dataset('ood_abstract', 100, 10)
    save_trigger_set(trigger_set, 'dataset/abstract_100')

