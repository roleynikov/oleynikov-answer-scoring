from pathlib import Path
import subprocess


def download_data(data: Path) -> None:
    if data.exists() and any(data.iterdir()):
        return

    subprocess.run(["dvc", "pull"], check=True)
