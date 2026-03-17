import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

NO_LABEL = -1

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


class CustomDataset(Dataset):
    """Example of a custom dataset, built from on labeled set and optionnal unlabeled set"""
    def __init__(self, labeled_data: str, unlabeled_data: str, transform):
        assert os.path.exists(labeled_data + "/inputs")
        assert os.path.exists(labeled_data + "/targets")
        if unlabeled_data:
            assert os.path.exists(unlabeled_data + "/inputs")
        
        self.n_l = len(os.listdir(labeled_data + "/inputs"))
        self.n_u = len(os.listdir(unlabeled_data + "/inputs")) if unlabeled_data else 0

        def load_img(path: str):
            from PIL import Image
            return Image.open(path)

        def load_label(path: str):
            f = open(path)
            content = f.read()
            f.close()
            return int(content[0])
        
        path_labeled_inputs = [labeled_data+"/inputs/"+p for p in os.listdir(labeled_data+"/inputs")] 
        path_unlabeled_inputs = [unlabeled_data+"/inputs/"+p for p in os.listdir(unlabeled_data+"/inputs")] if unlabeled_data else []
        path_targets = [labeled_data+"/targets/"+p for p in os.listdir(labeled_data+"/targets")]
        self.inputs = torch.stack([transform(load_img(path)) for path in path_labeled_inputs+path_unlabeled_inputs]) 
        self.targets = [load_label(path) for path in path_targets]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if idx < self.n_l:
            return self.inputs[idx], self.targets[idx]

        return self.inputs[idx], NO_LABEL


def load_custom_task(name, labeled_root, unlabeled_root, eval_root, **kwargs):
    """CustomDataset Instantiation"""
    train_ds = CustomDataset( 
        labeled_root,
        unlabeled_root,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    if len(train_ds) == 0:
        raise ValueError(f"Custom dataset is empty")

    mean_std = get_mean_std(train_ds)
    train_ds = CustomDataset( 
        labeled_root,
        unlabeled_root,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**mean_std)
        ])
    )
    eval_ds = CustomDataset(
        eval_root,
        None,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**mean_std)
        ])
    )

    labeled_indices, unlabeled_indices = list(range(0,train_ds.n_l)), list(range(train_ds.n_l, train_ds.n_l+train_ds.n_u))

    kwargs["train_ds"] = train_ds
    kwargs["eval_ds"] = eval_ds
    kwargs["labeled_indices"] = labeled_indices
    kwargs["unlabeled_indices"] = unlabeled_indices

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

    return kwargs



