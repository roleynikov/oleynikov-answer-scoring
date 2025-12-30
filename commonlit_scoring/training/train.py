import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import subprocess
from commonlit_scoring.training.lightning_module import CommonLitModel
from commonlit_scoring.data.datamodule import CommonLitDataModule
from dvc.repo import Repo


def train(cfg):
    Repo().pull()

    datamodule = CommonLitDataModule(cfg)
    model = CommonLitModel(cfg)
    logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
    )
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    logger.log_hyperparams({"git_commit": commit_hash})
    trainer.fit(model, datamodule=datamodule)
