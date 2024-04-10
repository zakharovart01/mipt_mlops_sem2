import os

import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

import hydra
import torch
from omegaconf import DictConfig

from scripts.dataset import Dataset
from scripts.model import ModelForClassification
from scripts.trainer import Trainer

torch.manual_seed(42)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Loading data
    PATH = cfg.params.PATH
    MODEL_PATH = cfg.params.MODEL_PATH
    MAX_LEN = cfg.params.MAX_LEN
    BATCH_SIZE = cfg.params.BATCH_SIZE
    train_data = pd.read_csv(os.path.join(PATH, "train.csv"))

    train_data = train_data.rename(columns={"label": "rate"})

    train_split, val_split = train_test_split(
        train_data, test_size=0.25, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "cointegrated/rubert-tiny2", truncation=True, do_lower_case=True
    )

    train_dataset = Dataset(train_split, tokenizer, MAX_LEN)
    val_dataset = Dataset(val_split, tokenizer, MAX_LEN)

    train_params = cfg.train_params
    test_params = cfg.test_params

    train_dataloader = DataLoader(train_dataset, **train_params)
    val_dataloader = DataLoader(val_dataset, **test_params)

    config = cfg.model

    model = ModelForClassification(cfg.params.HUG_FACE, config=config)

    trainer_config = {
        "lr": cfg.trainer.lr,
        "n_epochs": cfg.trainer.n_epochs,
        "weight_decay": cfg.trainer.weight_decay,
        "batch_size": BATCH_SIZE,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": cfg.params.SEED,
    }
    t = Trainer(trainer_config)

    t.fit(model, train_dataloader, val_dataloader)

    t.save(os.path.join(MODEL_PATH, "baseline_model.ckpt"))


if __name__ == "__main__":
    main()
