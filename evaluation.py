from sklearn.metrics import classification_report, accuracy_score
import torch
import json
import pickle
from clinicalBioBert import ClinicalDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import sys


def save_dict(filepath, metrics_dict):
  print("starting writing dictionary to a file")
  with open (filepath, 'w') as fp:
    json.dump(metrics_dict, fp)
  print("done writing dict into .txt file")

def prepare_data(data):
  tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
  labels = data['1'].tolist()
  labels = [1 if x == 'F' else 0 for x in labels]
  encodings = tokenizer(data['0'].tolist(), padding=True, truncation=True, max_length=128)
  tokens = ClinicalDataset(encodings, labels)
  loader = DataLoader(tokens, batch_size=8, shuffle=True)
  return loader, labels


def evaluate_model(modelpath, datapath, savepath):
  with open (datapath, 'rb') as f:
    df = pickle.load(f)

  loader, labels = prepare_data(df)
  model = torch.load(modelpath)
  preds = model(loader)
  metric_dict = classification_report(labels, preds, output_dict=True)
  save_dict(f'{savepath}_metrics.txt', metric_dict)

  return metric_dict


def main(modelpath, datapath, savepath):
  metric_dict = evaluate_model(modelpath, datapath, savepath)

if __name__ == "__main__":
  modelpath = sys.argv[1]
  datapath = sys.argv[2]
  savepath = sys.argv[3]
  main(modelpath, datapath, savepath)



