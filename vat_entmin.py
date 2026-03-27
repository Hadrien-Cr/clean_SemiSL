from omegaconf import DictConfig, ListConfig, OmegaConf
import hydra
from tqdm import tqdm
import wandb

import torch
import torchinfo
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn import functional as F
from torchvision.transforms import v2

from clean_SemiSL.utils.plot_utils import plot_batch
from clean_SemiSL.utils.schedulers import resolve_schedules 

NO_LABEL = -1

augment = v2.Compose([
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.AugMix()
])

def ce_loss(out_logits, target, label_smoothing=0.0):
    return F.cross_entropy(out_logits, target, label_smoothing=label_smoothing, reduction="mean")

def entropy(probs):
    return -(probs * probs.log().clamp(min=-1e7)).sum(dim=-1).mean()

def kl_div(out_probs, target_probs):
    return F.kl_div(out_probs.log(), target_probs, reduction="batchmean")

class InfiniteSampler(Sampler):
    def __init__(self, dataset_size, shuffle=True):
        self.dataset_size = dataset_size
        self.shuffle = shuffle

    def __iter__(self):
        while True:
            indices = torch.randperm(self.dataset_size) if self.shuffle \
                      else torch.arange(self.dataset_size)
            yield from indices.tolist()

    def __len__(self):
        return self.dataset_size  # nominal length

class SSLDataLoader:
    def __init__(self, unlabeled_dataset, labeled_dataset,
                 unlabeled_bs=256, labeled_bs=256, num_workers=4):

        self.unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=unlabeled_bs,
            sampler=InfiniteSampler(len(unlabeled_dataset)),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.labeled_loader = DataLoader(
            labeled_dataset,
            batch_size=labeled_bs,
            sampler=InfiniteSampler(len(labeled_dataset)),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self._unlabeled_iter = iter(self.unlabeled_loader)
        self._labeled_iter   = iter(self.labeled_loader)

    def __next__(self):
        (x_u, y_u) = next(self._unlabeled_iter)
        (x_l, y_l) = next(self._labeled_iter)
        return (x_u, y_u), (x_l, y_l)

    def __iter__(self):
        return self


@hydra.main(version_base=None, config_path="configs", config_name="vat_entmin")
def main(cfg): 
    hyperparams = cfg.hyperparams
    task = hydra.utils.instantiate(cfg.task)
    model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    task["metrics"] = {k: m.to(cfg.device) for k, m in task["metrics"].items()}
    
    not_bn_params = [p for n, p in model.named_parameters() if "bn" not in n and "norm" not in n]
    bn_params = [p for n, p in model.named_parameters() if "bn" in n or "norm" in n]

    lr = hyperparams.lr if isinstance(hyperparams.lr, float) else 0.0 # filled in the training loop
    regularization_coeff = hyperparams.regularization_coeff if isinstance(hyperparams.regularization_coeff, float) else 0.0 # filled in the training loop 
    
    if cfg.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            params=[
                {"params": not_bn_params},
                {"params": bn_params, "weight_decay": 0.0}
            ],
            lr=lr,
            weight_decay=cfg.optimizer.weight_decay,
            nesterov=cfg.optimizer.nesterov,
            momentum=cfg.optimizer.momentum
        )
    elif cfg.optimizer.name == "adam":
        optimizer = torch.optim.SGD(
            params=[
                {"params": not_bn_params},
                {"params": bn_params, "weight_decay": 0.0}
            ],
            lr=lr,
            weight_decay=cfg.optimizer.weight_decay,
        )

    if cfg.log_wandb:
        wandb.init(
            project=cfg.wandb.project_name,
            sync_tensorboard=True,
            config=OmegaConf.to_container(hyperparams), # type: ignore
            name=cfg.run_name,
        )
    writer = SummaryWriter(f"runs/{cfg.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" 
        for key, value in OmegaConf.to_container(hyperparams).items()])), #type: ignore
    )
    
    # Create DataLoaders
    print(f"#Labeled: {len(task['train_ds_labeled'])}; #Unlabeled: {len(task['train_ds_unlabeled'])}")

    eval_loader = DataLoader(
        task["eval_ds"],
        batch_size = hyperparams.batch_size,
        num_workers = hyperparams.workers,
        drop_last = False,
    )

    if not hyperparams.discard_unlabeled:
        train_loader = SSLDataLoader(
            unlabeled_dataset = task["train_ds_unlabeled"],
            labeled_dataset = task["train_ds_labeled"],
            unlabeled_bs=hyperparams.batch_size-hyperparams.labeled_batch_size,
            labeled_bs=hyperparams.labeled_batch_size,
            num_workers=hyperparams.workers
        )
    else:
         train_loader = iter(DataLoader(
            task["train_ds_labeled"],
            batch_size=hyperparams.batch_size,
            sampler=InfiniteSampler(len(task["train_ds_labeled"])),
            num_workers=hyperparams.workers,
            pin_memory=True,
            drop_last=True,
        ))

    torchinfo.summary(model, input_size=(1,)+task["train_ds_labeled"][0][0].shape, depth = 2)

    # Training Loop
    global_step = 0
    pbar = tqdm(range(hyperparams.num_iterations), disable = not cfg.verbose)

    for iteration in range(hyperparams.num_iterations):
        model.train()
        
        if isinstance(hyperparams.lr, DictConfig) and "schedules" in hyperparams.lr:
            lr = resolve_schedules(
                hyperparams.lr["schedules"], 
                t = iteration/hyperparams.num_iterations
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr 
        
        if isinstance(hyperparams.regularization_coeff, DictConfig) and "schedules" in hyperparams.regularization_coeff:
            regularization_coeff = resolve_schedules(
                hyperparams.regularization_coeff["schedules"], 
                t = iteration/hyperparams.num_iterations
            )

        if not hyperparams.discard_unlabeled:
            (x_u, _), (x_l,y_l) = next(train_loader)
            n_u = len(x_u)
            x = torch.cat([x_u, x_l], dim=0)
            x = x.to(cfg.device)
            y_l = y_l.to(cfg.device)

        else:
            x, y_l = next(train_loader)
            x = x.to(cfg.device)
            y_l = y_l.to(cfg.device)
            n_u = 0
        
        x = augment(x)
        out_logits = model(x)
         
        out_probs = F.softmax(out_logits, dim = -1) 
            
        supervision_loss = ce_loss(out_logits[n_u:],y_l, hyperparams.get("label_smoothing",0.0))
        
        if regularization_coeff > 0:
            # Compute VAT Loss
            xi, epsilon  = 1e-6, 2
            x_u = x[:n_u].detach()
            adversarial_perturbation = torch.Tensor(x_u.shape).normal_().to(cfg.device)
            adversarial_perturbation = xi * F.normalize(adversarial_perturbation)
            adversarial_perturbation.requires_grad_(True)

            out_perturb = model(x_u + adversarial_perturbation)
            divergence = kl_div(F.softmax(out_perturb, dim = -1), out_probs[:n_u].detach())
            divergence.backward()

            adversarial_perturbation = adversarial_perturbation.grad.data.clone()
            model.zero_grad()
            adversarial_perturbation = epsilon * F.normalize(adversarial_perturbation)
            out_perturb = model(x_u + adversarial_perturbation.detach())
            vat_loss = kl_div(F.softmax(out_perturb, dim = -1), out_probs[:n_u].detach())
            
            entmin_loss = entropy(out_probs[:n_u])
            regularization_loss = vat_loss + entmin_loss 
        else:
            regularization_loss = torch.tensor(0.0)

        loss = supervision_loss + regularization_coeff * regularization_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
        with torch.no_grad(): 
            out_probs_x_l = F.softmax(model(x[n_u:]), dim = -1)
        
        for k,m in task["metrics"].items():
            m(out_probs_x_l,y_l.float())
        
        pbar.set_description(f"Iter = {iteration+1} / {hyperparams.num_iterations}, LR = {lr:.5f}, RegCoeff = {regularization_coeff:.3f}, Loss = {loss.detach().item():.3f} " 
            + " ".join([f"{k}={m.compute():.3f}" for k,m in task["metrics"].items()]))
        pbar.update(1)

        global_step += hyperparams.batch_size
        writer.add_scalar("lr", lr, global_step = global_step)
        writer.add_scalar("regularization_coeff", regularization_coeff, global_step = global_step)
        writer.add_scalar("supervision_loss", supervision_loss.detach().item(), global_step = global_step)
        writer.add_scalar("regularization_loss", regularization_loss.detach().item(), global_step = global_step)
        writer.add_scalar("loss", loss.detach().item(), global_step = global_step)

        
        if (iteration + 1) % hyperparams.eval_frequency == 0: 
            for k,m in task["metrics"].items():
                writer.add_scalar("train/" + k, m.compute(), global_step = global_step)
                m.reset()

            model.eval()

            for x,y in eval_loader:
                x = x.to(cfg.device)
                y = y.to(cfg.device)
            
                with torch.no_grad():
                    out_probs_x = F.softmax(model(x), dim = -1) 

                for k,m in task["metrics"].items():
                    m(out_probs_x,y.float())
            
            for k,m in task["metrics"].items():
                writer.add_scalar("eval/" + k, m.compute(), global_step = global_step)
                m.reset()
        

    writer.close()
    if cfg.log_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
