from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    ds = hydra.utils.instantiate(cfg.task)
    print(ds)
    
if __name__ == "__main__":
    main()
