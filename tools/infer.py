import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd

from scripts.dataset import *
from scripts.model import *
from scripts.trainer import Trainer

def main():
    PATH = "data/"
    MAX_LEN = 128
    BATCH_SIZE = 1

    test_params = {"batch_size": BATCH_SIZE,
               "shuffle": False,
               "num_workers": 0
               }

    config = {
        "num_classes": 2,
        "dropout_rate": 0.1
    }
    model = ModelForClassification(
        "cointegrated/rubert-tiny2",
        config=config
    )

    tokenizer = AutoTokenizer.from_pretrained(
    "cointegrated/rubert-tiny2", truncation=True, do_lower_case=True)
    
    test_data = pd.read_csv(os.path.join(PATH, "test.csv"))
    test_data= test_data.rename(columns={'label': 'rate'})

    test_dataset = FiveDataset(test_data, tokenizer, MAX_LEN)
    test_dataloader = DataLoader(test_dataset, **test_params)

    t = Trainer.load("model/baseline_model.ckpt")
    test_data['pred_rate'] = t.predict(test_dataloader)

    test_data.to_csv('data/test_pred.csv')


if __name__ == "__main__":
    main()