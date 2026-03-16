import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

DATASETS = {
    "mnist": datasets.MNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
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

def load_task(name, root, download, **kwargs):
    """Dataset Instantiation""" 
    name = name.lower()
    
    train_dataset = DATASETS[name](
        root=root,
        download=download,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]) 
    )

    mean_std = get_mean_std(train_dataset)
    train_dataset = DATASETS[name](
        root=root,
        download=download,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**mean_std)
        ]) 
    )
    eval_dataset = DATASETS[name](
        root=root,
        download=download,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**mean_std)
        ]) 
    )
    
    kwargs["train_dataset"] = train_dataset
    kwargs["eval_dataset"] = eval_dataset 
    return kwargs

