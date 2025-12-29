from pathlib import Path
from omegaconf import DictConfig
from commonlit_scoring.data.load_data import download_data


def train(cfg: DictConfig) -> None:
    data = Path(cfg.data.data_dir)
    download_data(data)
    print("Данные загружены, обучение началось")
