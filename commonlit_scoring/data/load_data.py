from pathlib import Path
from dvc.repo import Repo


def download_data(data: Path) -> None:
    if data.exists() and any(data.iterdir()):
        return

    repo = Repo()
    repo.pull()
