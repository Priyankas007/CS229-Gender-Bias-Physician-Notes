import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import re

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class ClinicalDataset(Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
    return item

  def __len__(self):
    return len(self.labels)


# Make custon classifier model for ClinicalBioBERT
class ClinicalBioBERTClassifier(nn.Module):
    def __init__(self, pretrained_model_name="emilyalsentzer/Bio_ClinicalBERT", num_labels=2):
        super(ClinicalBioBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get the embeddings from ClinicalBioBERT
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        # The pooled output is usually used for classification tasks
        pooled_output = outputs.pooler_output

        # Pass pooled_output to classifier
        logits = self.classifier(pooled_output)

        # If labels are provided, calculate loss (useful for training)
        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return logits if loss is None else (loss, logits)


def make_tokens(csv_path, n=10000, filter=None):
  csv_file_path = csv_path
  df = pd.read_csv(csv_file_path)
  df = df.dropna()
  if filter == None:
    df['0'] = df['0'].fillna('').astype(str)
    df = df.head(n)
  else:
    df['0'] = df['0'].apply(replace_all, dic=filter)
    df['0'] = df['0'].fillna('').astype(str)
    df = df.head(n)

  # tokenize data
  tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

  labels = df['1'].tolist()
  labels = [1 if x == 'F' else 0 for x in labels]
  train_texts, val_texts, train_labels, val_labels = train_test_split(df['0'].tolist(), labels, test_size=0.2)
  train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=128)
  val_encodings = tokenizer(val_texts, padding=True, truncation=True, max_length=128)

  train_dataset = ClinicalDataset(train_encodings, train_labels)
  val_dataset = ClinicalDataset(val_encodings, val_labels)
  train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

  return train_loader, val_loader

def train_model(save_name, train_loader, val_loader):
  model = ClinicalBioBERTClassifier().to(device)
  optimizer = AdamW(model.parameters(), lr=5e-5)
  criterion = nn.CrossEntropyLoss().to(device)

  complete_train_loss = []
  complete_test_loss = []
  complete_train_acc = []
  complete_test_acc = []

  for epoch in tqdm.tqdm(range(3)):  # For demonstration, let's train for only 3 epochs
      model.train()
      i = 0
      total_train_loss = 0
      total_train_acc = 0

      for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        total_train_acc += (predictions == labels).sum().item()


      avg_train_loss = total_train_loss / len(train_loader)
      complete_train_loss.append(avg_train_loss)
      
      avg_train_acc = total_train_acc / len(train_loader.dataset)
      complete_train_acc.append(avg_train_acc)
      
      print(f'Train Loss: {avg_train_loss}, Train Accuracy: {avg_train_acc * 100}%')
      print(f"Epoch {epoch+1}, Loss: {loss.item()}")

      avg_test_loss, avg_test_acc = evaluate(model, val_loader, criterion)
      complete_test_loss.append(avg_test_loss)
      complete_test_acc.append(avg_test_acc)

  torch.save(model.state_dict(), f'{save_name}.pt')

  # save_dictionary
  final_dict = {
      'complete_train_loss' : complete_train_loss,
      'complete_train_acc' : complete_train_acc,
      'complete_test_loss' : complete_test_loss,
      'complete_test_acc' : complete_test_acc,
  }

  np.save(f'{save_name}.npy', final_dict)


def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_f1_acc = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            total_acc += (predictions == labels).sum().item()

    n = len(test_loader)
    avg_loss = total_loss / n
    avg_acc = total_acc / len(test_loader.dataset)
    print(f'Test Loss: {avg_loss}, Test Accuracy: {avg_acc * 100}%')
    return avg_loss, avg_acc

def f1_accuracy(preds, labels):
  pre = np.argmax(preds, axis=1).flatten()
  real = labels.flatten()
  return f1_score(real, pre)

def main(filepath):
  train_loader, val_loader = make_tokens(filepath, 1000)
  train_model('test', train_loader, val_loader)

if __name__ == "__main__":
  filepath = sys.argv[1]
  main(filepath)