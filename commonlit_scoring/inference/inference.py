from pathlib import Path
import pandas as pd
import torch

from commonlit_scoring.training.lightning_module import CommonLitModel


def infer(cfg):
    data_dir = Path(cfg.data.data_dir)
    test_path = data_dir / "test.csv"
    ckpt_path = Path(cfg.infer.checkpoint_path)
    out_path = Path(cfg.infer.output_path)

    assert test_path.exists(), "test.csv not found"
    assert ckpt_path.exists(), "checkpoint not found"

    model = CommonLitModel.load_from_checkpoint(ckpt_path, cfg=cfg, weights_only=False)
    model.eval()
    model.freeze()
    df = pd.read_csv(test_path)
    texts = df["text"].tolist()
    preds = []
    with torch.no_grad():
        for text in texts:
            pred = model.predict(text)
            preds.append(pred)

    df["prediction"] = preds
    df.to_csv(out_path, index=False)
    print(f"Предсказания сохранены в {out_path}")
