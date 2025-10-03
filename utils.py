import os
import numpy as np
import torch
from rdkit import Chem
from datetime import datetime

seed_value = 1
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rmse(y, f):
    return np.sqrt(((y - f) ** 2).mean(axis=0))

def mse(y, f):
    return ((y - f) ** 2).mean(axis=0)

def mse_cust(y, f):
    return ((y - f) ** 2)

def pearson(y, f):
    return np.corrcoef(y, f)[0, 1]

def spearman(y, f):
    from scipy import stats
    return stats.spearmanr(y, f)[0]

def ci(y, f):
    ind = np.argsort(y)
    y, f = y[ind], f[ind]
    i, S, z = len(y) - 1, 0.0, 0.0
    while i > 0:
        j = i - 1
        while j >= 0:
            if y[i] > y[j]:
                z += 1
                u = f[i] - f[j]
                if u > 0:
                    S += 1
                elif u == 0:
                    S += 0.5
            j -= 1
        i -= 1
    return S / z

def load_data():
    drug_smiles_file = 'data/processed/drug_features.npy'
    cell_data_file = 'data/processed/cell_data.npy'
    labels_file = 'data/labels.npy'
    if not os.path.isfile(drug_smiles_file):
        raise FileNotFoundError(f"Drug features file not found: {drug_smiles_file}")
    if not os.path.isfile(cell_data_file):
        raise FileNotFoundError(f"Cell data file not found: {cell_data_file}")
    if not os.path.isfile(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    drug_data = np.load(drug_smiles_file)
    cell_data = np.load(cell_data_file)
    labels = np.load(labels_file)
    return torch.tensor(drug_data, dtype=torch.float32), \
           torch.tensor(cell_data, dtype=torch.float32), \
           torch.tensor(labels, dtype=torch.float32)

def save_encoded_data(file_path, data):
    np.save(file_path, data)

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
