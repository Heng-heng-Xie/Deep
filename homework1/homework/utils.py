from PIL import Image
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here.
        """
        titles = []
        rows = []
        with open(dataset_path/'labels.csv', 'r') as file:
            data = csv.reader(file)
            titles = next(data)
            for i in data:
                rows.append(tuple(i[:1]))
        self.rows = rows
        raise NotImplementedError('SuperTuxDataset.__init__')

    def __len__(self):
        """
        Your code here
        """
        return len(self.rows)
        raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        return self.rows[idx]
        raise NotImplementedError('SuperTuxDataset.__getitem__')


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
