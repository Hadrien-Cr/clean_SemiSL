import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets

NO_LABEL = -1

DATASETS = {
    "mnist": datasets.MNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "imagenet": datasets.ImageNet
    #wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
    #wget https://mage-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
}


def get_mean_std(dataset):
    shape = dataset[0][0].shape
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
     
    mean = torch.zeros(shape[0])
    std = torch.zeros(shape[0])
    total_pixels = 0
    
    for images, _ in loader:
        batch_pixels = images.size(0) * images.size(2) * images.size(3)
        mean += images.sum([0, 2, 3]) / batch_pixels * batch_pixels
        std += (images ** 2).sum([0, 2, 3]) / batch_pixels * batch_pixels
        total_pixels += batch_pixels
    
    mean /= total_pixels
    std = torch.sqrt((std / total_pixels) - (mean ** 2))
    round_fn = lambda x: round(x,4)
    return dict(mean=list(map(round_fn, mean.tolist())), std=list(map(round_fn,std.tolist())))


class UnlabeledDataset(Dataset):
    def __init__(self, source_ds, indices):
        self.dataset = Subset(source_ds, indices)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, NO_LABEL * np.ones_like(y)

    def __len__(self):
        return len(self.dataset)


def relabel_dataset(source_ds: Dataset, labeling_ratio: int, seed: int):
    n = len(source_ds)
    rng_gen = np.random.default_rng(seed)
    labeled_indices = rng_gen.choice(n, int(n*labeling_ratio), replace=False)
    unlabeled_indices = [i for i in range(n) if i not in labeled_indices]
    
    return Subset(source_ds, labeled_indices), UnlabeledDataset(source_ds, unlabeled_indices)


def load_task(name, root, download, labeling_ratio, seed, **kwargs):
    """RelabeledDataset Instantiation""" 
    name = name.lower()
    
    train_ds = DATASETS[name](
        root=root,
        download=download,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]) 
    )

    mean_std = get_mean_std(train_ds)
    train_ds = DATASETS[name](
        root=root,
        download=download,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**mean_std)
        ]) 
    )
    eval_ds = DATASETS[name](
        root=root,
        download=download,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**mean_std)
        ]) 
    )

    train_ds_labeled, train_ds_unlabeled = relabel_dataset(
        train_ds, labeling_ratio, seed
    )

    kwargs["train_ds_labeled"] = train_ds_labeled
    kwargs["train_ds_unlabeled"] = train_ds_unlabeled
    kwargs["eval_ds"] = eval_ds
    
    def normalize_fn(tensor):
        mean = torch.tensor(mean_std["mean"], device=tensor.device).view(-1, 1, 1)
        std  = torch.tensor(mean_std["std"],  device=tensor.device).view(-1, 1, 1)
        return tensor.sub_(mean).div_(std)

    def denormalize_fn(tensor):
        mean = torch.tensor(mean_std["mean"], device=tensor.device).view(-1, 1, 1)
        std  = torch.tensor(mean_std["std"],  device=tensor.device).view(-1, 1, 1)
        return tensor.mul_(std).add_(mean)
    
    kwargs["normalize_fn"] = normalize_fn
    kwargs["denormalize_fn"] = denormalize_fn
    kwargs["metrics"] = {k: v for k, v in kwargs["metrics"].items()}
    return kwargs

