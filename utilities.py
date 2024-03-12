import json
import numpy as np


def load_data(xpath, ypath):
  with open(xpath, "r") as read_file:
      data = json.load(read_file)
      X = np.array(data["array"])

  sections_filtered = pd.read_csv(ypath)
  sections_filtered['tk'] = X.tolist()
  sections_filtered = sections_filtered.dropna() # remove all rows with NAN
  indices_kept = sections_filtered.index.tolist()

  X = sections_filtered['tk']
  Y = sections_filtered['1']

  return X, Y
