import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch




class DetectionSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=torchvision.transforms.ToTensor(), min_size=20):
        from glob import glob
        from os import path
        self.files = []
        self.labels = []
        for im_f in glob(path.join(dataset_path, 'images', '*')):
            self.files.append(im_f)

        for im_f in glob(path.join(dataset_path, 'labels', '*')):
            self.labels.append(im_f)
        self.transform = transform
        self.min_size = min_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import numpy as np
        b = self.files[idx]
        c = self.labels[idx]
        img = Image.open(b)
        label = Image.open(c)
        img = self.transform(img)
        label = self.transform(label)

        label = torch.clamp(torch.round(label), 0, 1)

        #  masks width
        width = torch.max(torch.sum(img, 1))
        width_label = label.clone()
        width_label[width_label == 1] = width

        return img, label.squeeze(0), width_label.squeeze(0)


def accuracy(pred, label):
    return (pred == label).float().mean().cpu().detach().numpy()


def load_detection_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DetectionSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)



