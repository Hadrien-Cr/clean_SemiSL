from omegaconf import DictConfig, OmegaConf
import hydra
from torch.utils.data import Dataset

def relabel_dataset(self, source_ds: Dataset, labelling_ratio: int, seed: int):
    n = len(source_ds)
    rng_gen = np.random.default_rng(seed)
    labelled = rng_gen.choice(n, int(n*labelling_ratio))
    labelled_indices, unlabelled_indices = [i for i in range(n) if labelled[i]], [i for i in range(n) if not labelled[i]]
    return labelled_indices, unlabelled_indices

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    ds = hydra.utils.instantiate(cfg.task)
    print(ds)
    
if __name__ == "__main__":
    main()
