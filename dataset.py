from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path


class ImageDataset(Dataset):

    def __init__(self, path, transform):
        dirs = sorted([dir for dir in Path(path).iterdir()])

        self.labels = [dir.name for dir in dirs] # class labels
        paths = np.array([sorted(list(Path(dir).iterdir())) for dir in dirs]).reshape(-1) # path of all images
        self.data = [(path, self.labels.index(path.parent.name)) for path in paths] # each data has image and label
                
        self.transform = transform

    def __getitem__(self, index):
        image_tensor = self.transform(Image.open(self.data[index][0]))
        label = self.data[index][1]
        return image_tensor, label

    def __len__(self):
        return len(self.data)