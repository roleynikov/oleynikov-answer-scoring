from pathlib import Path

from commonlit_scoring.data.load_data import download_data


def train() -> None:
    data = Path("data/raw")
    download_data(data)
    print("Данные загружены, обучение началось")
