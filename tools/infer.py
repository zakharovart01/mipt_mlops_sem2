import os

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import hydra
from omegaconf import DictConfig

from scripts.dataset import Dataset
from scripts.trainer import Trainer


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    PATH = cfg.params.PATH
    MODEL_PATH = cfg.params.MODEL_PATH
    MAX_LEN = cfg.params.MAX_LEN

    test_params = cfg.test_params

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.params.HUG_FACE, truncation=True, do_lower_case=True
    )

    test_data = pd.read_csv(os.path.join(PATH, "test.csv"))
    test_data = test_data.rename(columns={"label": "rate"})

    test_dataset = Dataset(test_data, tokenizer, MAX_LEN)
    test_dataloader = DataLoader(test_dataset, **test_params)

    t = Trainer.load(os.path.join(MODEL_PATH, "baseline_model.ckpt"))
    test_data["pred_rate"] = t.predict(test_dataloader)

    test_data.to_csv(os.path.join(PATH, "test_pred.csv"))


if __name__ == "__main__":
    main()
