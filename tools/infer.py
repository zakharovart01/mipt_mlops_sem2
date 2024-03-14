import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        logits = self.linear(x)
        return logits

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

def collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])

    return input_ids, attention_mask

def compute_average_embeddings(model, data_loader, device):
    all_embeddings = []
    model.eval()

    with torch.no_grad():
        for input_ids, attention_mask in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            embeddings = model(input_ids, attention_mask=attention_mask)
            embeddings = embeddings[0] * attention_mask.unsqueeze(-1)
            embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

            all_embeddings.append(embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

def predict(model, clf, texts, tokenizer, max_len, device):
    dataset = TextDataset(texts, tokenizer, max_len)
    data_loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

    X = compute_average_embeddings(model, data_loader, device)
    logits = clf(X)
    probs = torch.sigmoid(logits)
    y_pred = (probs >= 0.5).long()

    return y_pred.cpu().numpy()

def main():
    df = pd.read_csv('../../data/test.csv')[:5]
    texts = df["title"].to_numpy()
    # texts = ["example text 1", "example text 2", "example text 3"]

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    max_len = 128

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = AutoModel.from_pretrained('bert-base-uncased')
    model.to(device)

    # Load the PyTorch logistic regression model from a file
    clf = torch.load('../../model/model_cls.pt')

    y_pred = predict(model, clf, texts, tokenizer, max_len, device)
    data = pd.DataFrame()
    data["title"] = texts 
    data["pred"] = y_pred
    data.to_csv('../../data/test_pred.csv')

if __name__ == "__main__":
    main()