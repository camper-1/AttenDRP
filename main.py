import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from models import EnhancedDrugResponsePredictor
from train import train_model

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_raw_features():
    cn_features = np.load('./data/processed/cn_encoded.npy')
    exp_features = np.load('./data/processed/exp_encoded.npy')
    mu_features = np.load('./data/processed/mu_encoded.npy')
    cell_features = np.concatenate((cn_features, exp_features, mu_features), axis=1)

    fingerprint_features = np.load('./data/processed/fingerprint_features.npz')['fingerprints']
    physico_features = np.load('./data/processed/physicochemical_features.npz')['physico_features']
    transformed_smiles_features = np.load('./data/processed/transformed_smiles_features.npz')['smiles_features']

    fingerprint_features = fingerprint_features.reshape(fingerprint_features.shape[0], -1)
    transformed_smiles_features = transformed_smiles_features.reshape(transformed_smiles_features.shape[0], -1)
    drug_features = np.concatenate((fingerprint_features, physico_features, transformed_smiles_features), axis=1)

    drug_ic50_data = pd.read_csv('./data/drug/IC50.csv')
    drug_names = drug_ic50_data['Drug name'].unique()
    cell_names = drug_ic50_data['Cell line name'].unique()
    drug_to_idx = {name: idx for idx, name in enumerate(drug_names)}
    cell_to_idx = {name: idx for idx, name in enumerate(cell_names)}

    fused_drug_features, fused_cell_features, labels = [], [], []
    for _, row in drug_ic50_data.iterrows():
        d_idx = drug_to_idx[row['Drug name']]
        c_idx = cell_to_idx[row['Cell line name']]
        fused_drug_features.append(drug_features[d_idx])
        fused_cell_features.append(cell_features[c_idx])
        labels.append(row['IC50'])

    return np.array(fused_drug_features), np.array(fused_cell_features), np.array(labels)

def main():
    set_random_seed(2024)
    fused_drug_features, fused_cell_features, labels = load_raw_features()

    X_train_drug, X_temp_drug, X_train_cell, X_temp_cell, y_train, y_temp = train_test_split(
        fused_drug_features, fused_cell_features, labels, test_size=0.2, random_state=42
    )
    X_val_drug, X_test_drug, X_val_cell, X_test_cell, y_val, y_test = train_test_split(
        X_temp_drug, X_temp_cell, y_temp, test_size=0.5, random_state=42
    )

    drug_scaler = StandardScaler().fit(X_train_drug)
    cell_scaler = StandardScaler().fit(X_train_cell)

    X_train_drug = drug_scaler.transform(X_train_drug)
    X_val_drug = drug_scaler.transform(X_val_drug)
    X_test_drug = drug_scaler.transform(X_test_drug)

    X_train_cell = cell_scaler.transform(X_train_cell)
    X_val_cell = cell_scaler.transform(X_val_cell)
    X_test_cell = cell_scaler.transform(X_test_cell)

    X_train_drug = torch.tensor(X_train_drug, dtype=torch.float32)
    X_val_drug = torch.tensor(X_val_drug, dtype=torch.float32)
    X_test_drug = torch.tensor(X_test_drug, dtype=torch.float32)

    X_train_cell = torch.tensor(X_train_cell, dtype=torch.float32)
    X_val_cell = torch.tensor(X_val_cell, dtype=torch.float32)
    X_test_cell = torch.tensor(X_test_cell, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train_drug, X_train_cell, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_drug, X_val_cell, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_drug, X_test_cell, y_test), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EnhancedDrugResponsePredictor(
        drug_feature_dim=X_train_drug.shape[1],
        cell_feature_dim=X_train_cell.shape[1],
        hidden_dim=512,
        num_heads=8,
        dropout_rate=0.3
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    metrics = train_model(
        model,
        train_loader,
        val_loader,
        device,
        optimizer,
        scheduler,
        criterion,
        num_epochs=300,
        patience=20,
        label='DrugResponse'
    )

    print("Training finished. Ready for evaluation.")

if __name__ == "__main__":
    main()
