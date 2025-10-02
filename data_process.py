import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import Autoencoder, MLPModel, CNNModel, SMILESTransformer
from drug_feature import smiles_to_fingerprint, smiles_to_physicochemical_properties
from utils import is_valid_smiles
from rdkit import Chem

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(file_path, num_columns):
    data = pd.read_csv(file_path, sep=',', header=None)
    data = data.iloc[:, 1:]
    numeric_data = data.apply(pd.to_numeric, errors='coerce').fillna(0).values
    return numeric_data[:, :num_columns]

def train_autoencoder(input_data, hidden_dim, num_epochs, lr):
    model = Autoencoder(input_dim=input_data.shape[1], hidden_dim=hidden_dim).to(device)
    data_loader = DataLoader(torch.tensor(input_data, dtype=torch.float32).to(device), batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    for _ in range(num_epochs):
        epoch_loss = 0
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step(epoch_loss / len(data_loader))
    model.eval()
    with torch.no_grad():
        encoded = model.encoder_layer2(model.encoder_layer1(torch.tensor(input_data, dtype=torch.float32).to(device))).cpu().numpy()
    return model, encoded

def encode_with_model(model, data, scaler):
    scaled = scaler.transform(data)
    with torch.no_grad():
        encoded = model.encoder_layer2(model.encoder_layer1(torch.tensor(scaled, dtype=torch.float32).to(device))).cpu().numpy()
    return encoded

def process_cell_line_data():
    cell_cn = load_data('data/cell_line/cn_580cell_706gene.csv', 706)
    cell_exp = load_data('data/cell_line/exp_580cell_706gene.csv', 706)
    cell_mu = load_data('data/cell_line/mu_580cell_706gene.csv', 706)

    drug_ic50 = pd.read_csv('data/drug/IC50.csv')
    cell_names = drug_ic50['Cell line name'].unique()
    idx_map = {name: i for i, name in enumerate(cell_names)}

    fused_indices = [idx_map[name] for name in drug_ic50['Cell line name']]
    fused_features = np.stack([np.concatenate([cell_cn[i], cell_exp[i], cell_mu[i]]) for i in fused_indices])

    X_train, X_temp = train_test_split(fused_features, test_size=0.2, random_state=42)
    X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

    scaler = MinMaxScaler().fit(X_train)

    cn_train = scaler.transform(cell_cn)
    exp_train = scaler.transform(cell_exp)
    mu_train = scaler.transform(cell_mu)

    cn_model, _ = train_autoencoder(cn_train, hidden_dim=256, num_epochs=100, lr=0.001)
    exp_model, _ = train_autoencoder(exp_train, hidden_dim=256, num_epochs=100, lr=0.001)
    mu_model, _ = train_autoencoder(mu_train, hidden_dim=256, num_epochs=100, lr=0.001)

    save_dir = './data/processed'
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'cn_encoded.npy'), encode_with_model(cn_model, cell_cn, scaler))
    np.save(os.path.join(save_dir, 'exp_encoded.npy'), encode_with_model(exp_model, cell_exp, scaler))
    np.save(os.path.join(save_dir, 'mu_encoded.npy'), encode_with_model(mu_model, cell_mu, scaler))

def process_drug_data():
    drug_smiles_file = 'data/drug/drug_smiles.csv'
    fingerprint_output_file = 'data/processed/fingerprint_features.npz'
    physico_output_file = 'data/processed/physicochemical_features.npz'
    transformed_smiles_output_file = 'data/processed/transformed_smiles_features.npz'

    drug_data = pd.read_csv(drug_smiles_file)
    smiles_list = drug_data['Isosmiles'].values

    unique_chars = set("".join(smiles_list))
    unique_chars.update('cno')
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    char_to_idx['<PAD>'] = len(char_to_idx)
    vocab_size = len(char_to_idx)

    cnn_model = CNNModel(input_dim=1, hidden_dim=128, reduced_dim=128).to(device)
    mlp_model = MLPModel(input_dim=9, hidden_dim=128, output_dim=128).to(device)
    transformer_model = SMILESTransformer(vocab_size=vocab_size, embed_dim=256, num_heads=4, num_layers=3).to(device)

    max_length = 100
    aug_num = 6

    all_fingerprint_features = []
    all_physico_features = []
    all_transformed_smiles_features = []

    for smiles in smiles_list:
        aug_smiles_set = {Chem.MolToSmiles(Chem.MolFromSmiles(smiles), doRandom=True) for _ in range(aug_num)}
        aug_smiles_set.add(smiles)

        batch_smiles_tensors = []
        batch_fingerprints = []
        batch_physico = []

        for aug_smiles in aug_smiles_set:
            if is_valid_smiles(aug_smiles):
                smiles_indices = [char_to_idx.get(char, char_to_idx['<PAD>']) for char in aug_smiles]
                smiles_indices = smiles_indices[:max_length] + [char_to_idx['<PAD>']] * (max_length - len(smiles_indices))
                smiles_tensor = torch.tensor(smiles_indices, dtype=torch.long)
                batch_smiles_tensors.append(smiles_tensor)

                fingerprint = smiles_to_fingerprint(aug_smiles)
                batch_fingerprints.append(fingerprint)
                physicochemical_properties = smiles_to_physicochemical_properties(aug_smiles)
                batch_physico.append(physicochemical_properties)

        if batch_smiles_tensors:
            batch_smiles_tensors = torch.stack(batch_smiles_tensors).to(device)
            batch_fingerprint_tensors = torch.tensor(batch_fingerprints, dtype=torch.float32).unsqueeze(1).to(device)
            batch_physico_tensors = torch.tensor(batch_physico, dtype=torch.float32).to(device)

            transformed_outputs = transformer_model(batch_smiles_tensors).cpu().detach().numpy()
            cnn_outputs = cnn_model(batch_fingerprint_tensors).cpu().detach().numpy()
            physico_outputs = mlp_model(batch_physico_tensors).cpu().detach().numpy()

            all_fingerprint_features.extend(cnn_outputs)
            all_physico_features.extend(physico_outputs)
            all_transformed_smiles_features.extend(transformed_outputs)

    np.savez_compressed(fingerprint_output_file, fingerprints=np.array(all_fingerprint_features))
    np.savez_compressed(physico_output_file, physico_features=np.array(all_physico_features))
    np.savez_compressed(transformed_smiles_output_file, smiles_features=np.array(all_transformed_smiles_features))

if __name__ == "__main__":
    process_cell_line_data()
    process_drug_data()
