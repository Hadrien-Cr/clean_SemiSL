import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

class SemiLabelledDataset(Dataset):
    def __init__(self, source_ds: Dataset, labelling_ratio: int, seed: int, **kwargs):
        self.source_ds = source_ds
        rng_gen = np.random.default_rng(seed)
        self.labelled_indices = rng_gen.choice(len(self.source_ds), int(len(self.source_ds)*labelling_ratio))
        self.label_count = int(len(self.source_ds)*labelling_ratio)
        self.kwargs = kwargs
    
    def __len__(self):
        return len(self.source_ds)

    def __getitem__(self, idx):
        x,y = super.__getitem__(idx)
        if self.labbelled_indices(idx):
            return x,y
        return x, None
