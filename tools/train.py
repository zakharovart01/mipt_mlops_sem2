import os

import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from scripts.dataset import *
from scripts.model import *
from scripts.trainer import Trainer
torch.manual_seed(42)


def main():
    # Loading data
    PATH = "data/"
    MAX_LEN = 128
    BATCH_SIZE = 1
    train_data = pd.read_csv(os.path.join(PATH, "train.csv"))

    train_data = train_data.rename(columns={'label': 'rate'})
    # le = LabelEncoder()
    # train_data.rate = le.fit_transform(train_data.rate)
    
    train_split, val_split = train_test_split(train_data, test_size=0.25, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(
    "cointegrated/rubert-tiny2", truncation=True, do_lower_case=True)

    train_dataset = FiveDataset(train_split, tokenizer, MAX_LEN)
    val_dataset = FiveDataset(val_split, tokenizer, MAX_LEN)

    train_params = {"batch_size": BATCH_SIZE,
                "shuffle": True,
                "num_workers": 0
                }

    test_params = {"batch_size": BATCH_SIZE,
               "shuffle": False,
               "num_workers": 0
               }

    train_dataloader = DataLoader(train_dataset, **train_params)
    val_dataloader = DataLoader(val_dataset, **test_params)

    config = {
    "num_classes": 2,
    "dropout_rate": 0.1
    }
    model = ModelForClassification(
        "cointegrated/rubert-tiny2",
        config=config
    )

    trainer_config = {
    "lr": 3e-4,
    "n_epochs": 2,
    "weight_decay": 1e-6,
    "batch_size": BATCH_SIZE,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    }
    t = Trainer(trainer_config)

    t.fit(
        model,
        train_dataloader,
        val_dataloader
    )

    t.save("model/baseline_model.ckpt")


if __name__ == "__main__":
    main()