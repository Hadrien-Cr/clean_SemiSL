import itertools
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm
import wandb
import math

import numpy as np
import torch
import torchinfo
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler, SubsetRandomSampler
from torch.nn import functional as F
from torchvision.transforms import v2

from clean_SemiSL.utils.plot_utils import plot_batch

NO_LABEL = -1

augment = v2.AugMix()

def linear_schedule(initial_value, final_value, t, t_start, t_end):
    progress = np.clip((t-t_start)/(t_end-t_start), 0, 1)
    return initial_value + (final_value - initial_value) * progress

def cosine_schedule(initial_value, final_value, t, t_start, t_end):
    progress = np.clip((t-t_start)/(t_end-t_start), 0, 1)
    return final_value + (initial_value - final_value) * (1 + np.cos(np.pi * progress)) / 2 

def rescale_probs(probs, temperature):
    logits = torch.log(probs.clamp(min=1e-7))
    scaled = F.softmax(logits / temperature, dim=-1)
    return scaled.clamp(min=1e-9, max=1 - 1e-9) 

def kl_div_loss(out_probs, target_probs):
    log_pred = torch.log(out_probs.clamp(min=1e-7))
    return F.kl_div(log_pred, target_probs, reduction="batchmean")

def bce_loss(out_probs, target_probs):
    out_probs = out_probs.clamp(min=1e-7, max=1 - 1e-7)
    return F.binary_cross_entropy(out_probs, target_probs)

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices: primary (unlabeled) and secondary (labeled)
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
    for i in labeled_indices:
        (x,y) = train_ds[i]
        print(y)
        if i > 1000: break

    if discard_unlabeled:
        subset_sampler = SubsetRandomSampler(labeled_indices)
        batch_sampler = BatchSampler(subset_sampler, batch_size, drop_last=False)
    else:
        batch_sampler = TwoStreamBatchSampler(
            unlabeled_indices, 
            labeled_indices, 
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
    hyperparams = cfg.hyperparams
    task = hydra.utils.instantiate(cfg.task)
    model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    task["metrics"] = {k: m.to(cfg.device) for k, m in task["metrics"].items()}
    
    if cfg.log_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb.project_name,
            sync_tensorboard=True,
            config=OmegaConf.to_container(hyperparams),
            name=cfg.run_name,
        )
    writer = SummaryWriter(f"runs/{cfg.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in OmegaConf.to_container(hyperparams).items()])),
    )

    train_loader, eval_loader = create_dataloaders(
        task["train_ds"],
        task["eval_ds"],
        discard_unlabeled=hyperparams.discard_unlabeled,
        batch_size=hyperparams.batch_size,
        labeled_batch_size=hyperparams.labeled_batch_size,
        labeled_indices=task["labeled_indices"],
        unlabeled_indices=task["unlabeled_indices"],
        workers=hyperparams.workers
    )

    optimizer = torch.optim.Adam(
        model.parameters(), 
        hyperparams.lr.v0,
        weight_decay=hyperparams.weight_decay
    )

    torchinfo.summary(model, input_size=(1,)+task["train_ds"][0][0].shape, depth = 2)

    # Training Loop
    global_step = 0

    for epoch in range(hyperparams.num_epochs):
        
        model.train()

        pbar = tqdm(train_loader, disable = not cfg.verbose)
        num_batches = len(pbar)

        for i, (x,y) in enumerate(pbar):
            lr = cosine_schedule(
                hyperparams.lr.v0, 
                hyperparams.lr.vf, 
                epoch + i/num_batches, 
                hyperparams.lr.ramp_start,
                hyperparams.lr.ramp_end
            )
            regularization_coeff = linear_schedule(
                hyperparams.regularization_coeff.v0, 
                hyperparams.regularization_coeff.vf, 
                epoch + i /num_batches,
                hyperparams.regularization_coeff.ramp_start,
                hyperparams.regularization_coeff.ramp_end
            )

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr 

            x = x.to(cfg.device)
            y = y.to(cfg.device)

            l_indices = torch.ne(y, NO_LABEL).nonzero(as_tuple=True)
            u_indices = torch.eq(y, NO_LABEL).nonzero(as_tuple=True)

            x1, x2 = augment(x), augment(x)
            out_logits_x1 = model(x1)
            
            with torch.no_grad(): out_logits_x2 = model(x2)
            
            out_probs_x1, out_probs_x2 = F.softmax(out_logits_x1, dim = -1), F.softmax(out_logits_x2, dim = -1) 
            
            supervision_loss = bce_loss(
                out_probs_x1[l_indices],
                F.one_hot(y[l_indices], task["num_classes"]).float()
            )

            pseudo_label = out_probs_x2.argmax(-1).detach()

            regularization_loss = kl_div_loss(
                out_probs_x1,
                rescale_probs(out_probs_x2, hyperparams.sharpening_temperature)
            )

            loss = supervision_loss + regularization_coeff * regularization_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
  
            with torch.no_grad():
                out_logits = model(x[l_indices])
            
            pred = F.softmax(out_logits, dim=-1)

            for k,m in task["metrics"].items():
                m(pred,y[l_indices].float())

            pbar.set_description(f"Epoch = {epoch+1}/{hyperparams.num_epochs}, LR = {lr:.5f}, Loss = {loss.detach().item():.3f} " + " ".join([f"{k}={m.compute():.3f}" for k,m in task["metrics"].items()]))
            
            global_step += hyperparams.batch_size
            writer.add_scalar("lr", lr, global_step = global_step)
            writer.add_scalar("regularization_coeff", regularization_coeff, global_step = global_step)
            writer.add_scalar("supervision_loss", supervision_loss.detach().item(), global_step = global_step)
            writer.add_scalar("regularization_loss", regularization_loss.detach().item(), global_step = global_step)
            writer.add_scalar("loss", loss.detach().item(), global_step = global_step)

        for k,m in task["metrics"].items():
            writer.add_scalar("train/" + k, m.compute(), global_step = global_step)
            m.reset()

        model.eval()

        for x,y in tqdm(eval_loader, disable = not cfg.verbose):
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            
            with torch.no_grad():
                out_logits = model(x)
            
            pred = F.softmax(out_logits, dim = -1) 

            for k,m in task["metrics"].items():
                m(pred,y.float())

            pbar.set_description(f"Eval Metrics: " + " ".join([f"{k}={m.compute():.3f}" for k,m in task["metrics"].items()]))
            
        for k,m in task["metrics"].items():
            writer.add_scalar("eval/" + k, m.compute(), global_step = global_step)
            m.reset()
        
    writer.close()
    if cfg.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
