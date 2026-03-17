import itertools
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler, SubsetRandomSampler

from clean_SemiSL.utils.plot_utils import plot_batch

NO_LABEL = -1

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices: primary (labeled) and secondary (unlabeled)
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = self.iterate_once(self.primary_indices)
        secondary_iter = self.iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(self.grouper(primary_iter, self.primary_batch_size),
                    self.grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

    def iterate_once(self, indices):
        return np.random.permutation(indices)

    def iterate_eternally(self, indices):
        def infinite_shuffles():
            while True:
                yield self.iterate_once(indices)
        return itertools.chain.from_iterable(infinite_shuffles())

    def grouper(self, iterable, n):
        "Collect data into fixed-length chunks or blocks"
        # grouper('ABCDEFG', 3) --> ABC DEF"
        args = [iter(iterable)] * n
        return zip(*args)


def create_dataloaders(
    train_ds: Dataset,
    eval_ds: Dataset,
    discard_unlabeled: bool,
    labeled_batch_size: int,
    batch_size: int,
    labeled_indices: list[int],
    unlabeled_indices: list[int],
    workers: int,
):
    if discard_unlabeled:
        subset_sampler = SubsetRandomSampler(labeled_indices)
        batch_sampler = BatchSampler(subset_sampler, batch_size)
    else:
        batch_sampler = TwoStreamBatchSampler(
            labeled_indices, 
            unlabeled_indices, 
            batch_size=batch_size,
            secondary_batch_size=labeled_batch_size
        )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=True
    )
    eval_loader = DataLoader(
        dataset=eval_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=workers
    )
    return train_loader, eval_loader


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    task = hydra.utils.instantiate(cfg.task)
    
    train_loader, eval_loader = create_dataloaders(
        task["train_ds"],
        task["eval_ds"],
        discard_unlabeled=False,
        batch_size=cfg.batch_size,
        labeled_batch_size=cfg.labeled_batch_size,
        labeled_indices=task["labeled_indices"],
        unlabeled_indices=task["unlabeled_indices"],
        workers=cfg.workers
    )
    
    for x, y in train_loader:
        unnormalized_x = task["denormalize_fn"](x)
        plot_batch(
            unnormalized_x,
            None,
            y,
            task_type=task["type"],
            save="here.png"
        )
        break
    
if __name__ == "__main__":
    main()









