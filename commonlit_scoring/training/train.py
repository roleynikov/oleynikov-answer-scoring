import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import subprocess
from commonlit_scoring.training.lightning_module import CommonLitModel
from commonlit_scoring.data.datamodule import CommonLitDataModule
from dvc.repo import Repo
from pathlib import Path


def train(cfg):
    Repo().pull()

    datamodule = CommonLitDataModule(cfg)
    model = CommonLitModel(cfg)
    logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=Path(cfg.trainer.output_dir),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_cb],
    )

    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    logger.log_hyperparams({"git_commit": commit_hash})
    trainer.fit(model, datamodule=datamodule)
    print(f"Лучшая модель сохранена в : {checkpoint_cb.best_model_path}")
