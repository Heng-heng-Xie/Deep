from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    """
    WARNING: Do not perform data normalization here. 
    """
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use your solution (or the master solution) to HW1
        """
        from os import path
        import csv
        self.rows = []
        image_to_tensor = transforms.ToTensor()
        with open(path.join(dataset_path, 'labels.csv'), newline='') as file:
            data = csv.reader(file)
            for f, l, _ in data:
                if l in LABEL_NAMES:
                    im = Image.open(path.join(dataset_path, f))
                    label_n = LABEL_NAMES.index(l)
                    self.rows.append((image_to_tensor(im), label_n))



    def __len__(self):
        """
        Your code here
        """
        return len(self.rows)


    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        return self.rows[idx]



def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
