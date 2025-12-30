import hydra
from omegaconf import DictConfig
from commonlit_scoring.training.train import train
from commonlit_scoring.inference.inference import infer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg.mode)
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "infer":
        infer(cfg)
    else:
        raise ValueError("Введите mode=[train/infer]")
