import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd

torch.manual_seed(42)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

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
            'label': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    labels = torch.stack([x['label'] for x in batch])

    return input_ids, attention_mask, labels

def compute_average_embeddings(model, data_loader, device):
    all_embeddings = []
    model.eval()

    with torch.no_grad():
        for input_ids, attention_mask, _ in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            embeddings = model(input_ids, attention_mask=attention_mask)
            embeddings = embeddings[0] * attention_mask.unsqueeze(-1)
            embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

            all_embeddings.append(embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        logits = self.linear(x)
        return logits

def train_logistic_regression(X_train, y_train, X_val, y_val, input_dim, output_dim, device):
    model = LogisticRegression(input_dim, output_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    best_val_loss = float('inf')
    epochs = 10

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train)
        loss = criterion(logits.squeeze(), torch.tensor(y_train).float())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits.squeeze(), torch.tensor(y_val).float().squeeze())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

    return best_model

def main():
    # Load your dataset here (for example, using pandas)
    df = pd.read_csv('../../data/train.csv')[:15]
    texts = df["title"].to_numpy()
    labels = df["label"].to_numpy()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    max_len = 128

    dataset = TextDataset(texts, labels, tokenizer, max_len)
    data_loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = AutoModel.from_pretrained('bert-base-uncased')
    model.to(device)

    X = compute_average_embeddings(model, data_loader, device)

    train_val_split = int(0.8 * len(X))
    X_train, X_val = X[:train_val_split], X[train_val_split:]
    y_train, y_val = labels[:train_val_split], labels[train_val_split:]

    input_dim = X_train.shape[1]
    output_dim = 1

    model_cls = train_logistic_regression(X_train, y_train, X_val, y_val, input_dim, output_dim, device)
    torch.save(model_cls, '../../model/model_cls.pt')

if __name__ == "__main__":
    main()