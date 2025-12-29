import hydra
from omegaconf import DictConfig
from commonlit_scoring.training.train import train


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)
