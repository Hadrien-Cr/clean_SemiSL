from torchvision import datasets
from ..utils.semi_labelled_data import SemiLabelledDataset

DATASETS = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "imagenet": datasets.ImageNet
}

def load_dataset(name, **kwargs):
    name = name.lower()
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    source_ds = DATASETS[name](**kwargs)
    return SemiLabelledDataset(source_ds=source_ds,**kwargs)

