import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch_geometric.nn as pyg_nn  # GNN相关库
from torch.utils.data import DataLoader, Dataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder_layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        self.adjust_dim = nn.Linear(input_dim, hidden_dim // 2)
        self.decoder_layer1 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.decoder_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = self.adjust_dim(x)
        encoded = self.encoder_layer1(x)
        encoded = self.encoder_layer2(encoded) + residual
        decoded = self.decoder_layer1(encoded)
        decoded = self.decoder_layer2(decoded)
        return decoded



class CNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, reduced_dim=128):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, stride=2, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc = None
        self.reduced_dim = reduced_dim
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.flatten(x)
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), self.reduced_dim).to(x.device)
        x = self.fc(x)
        return x




class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        return x



class FingerprintAttention(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_heads=8, dropout=0.1):
        super(FingerprintAttention, self).__init__()


        self.fc = nn.Linear(input_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.residual_fc = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, fingerprint_features):

        projected_features = self.fc(fingerprint_features)


        projected_features = self.layer_norm1(projected_features)

        attn_output, _ = self.attention(projected_features.unsqueeze(0),
                                        projected_features.unsqueeze(0),
                                        projected_features.unsqueeze(0))
        attn_output = attn_output.squeeze(0)
        attn_output = self.dropout(attn_output)


        attn_output = self.layer_norm2(attn_output)


        residual_input = self.residual_fc(fingerprint_features)
        output = attn_output + residual_input

        return output


class EnhancedDrugResponsePredictor(nn.Module):
    def __init__(self, drug_feature_dim, cell_feature_dim, hidden_dim=256, num_heads=8, dropout_rate=0.3,
                 attention_layers=3):
        super(EnhancedDrugResponsePredictor, self).__init__()
        self.drug_fc1 = nn.Linear(drug_feature_dim, hidden_dim)
        self.cell_fc1 = nn.Linear(cell_feature_dim, hidden_dim)
        self.drug_bn1 = nn.BatchNorm1d(hidden_dim)
        self.cell_bn1 = nn.BatchNorm1d(hidden_dim)

        self.cross_attentions_drug_to_cell = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
            for _ in range(attention_layers)
        ])
        self.cross_attentions_cell_to_drug = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
            for _ in range(attention_layers)
        ])


        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, drug_data, cell_data):

        drug_out = F.relu(self.drug_bn1(self.drug_fc1(drug_data)))
        cell_out = F.relu(self.cell_bn1(self.cell_fc1(cell_data)))
        drug_out = drug_out.unsqueeze(1)  # (N, 1, D)
        cell_out = cell_out.unsqueeze(1)  # (N, 1, D)


        for cross_attention_drug_to_cell, cross_attention_cell_to_drug in zip(
                self.cross_attentions_drug_to_cell, self.cross_attentions_cell_to_drug
        ):

            attn_drug_to_cell, _ = cross_attention_drug_to_cell(drug_out, cell_out, cell_out)
            drug_out = attn_drug_to_cell + drug_out


            attn_cell_to_drug, _ = cross_attention_cell_to_drug(cell_out, drug_out, drug_out)
            cell_out = attn_cell_to_drug + cell_out


        combined = torch.cat([drug_out.squeeze(1), cell_out.squeeze(1)], dim=1)


        combined = self.dropout(F.relu(self.bn2(self.fc2(combined))))
        combined = self.dropout(F.relu(self.bn3(self.fc3(combined))))
        out = self.fc4(combined)
        return out

    def forward_features(self, drug_data, cell_data):
        """提取药物和细胞的特征表示"""
        drug_out = F.relu(self.drug_bn1(self.drug_fc1(drug_data)))
        cell_out = F.relu(self.cell_bn1(self.cell_fc1(cell_data)))
        return drug_out, cell_out

class SMILESTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout_rate=0.1, max_length=100, pad_idx=0):
        super(SMILESTransformer, self).__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, embed_dim))  # 可学习的位置编码
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate),
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        pad_mask = (x == self.pad_idx)
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        transformer_out = self.transformer_encoder(
            x.permute(1, 0, 2),
            src_key_padding_mask=pad_mask
        )
        out = self.output_layer(transformer_out.permute(1, 0, 2))
        return self.dropout(out)



class SMILESDataset(Dataset):
    def __init__(self, smiles_list, char_to_idx, max_length, unknown_token='<UNK>', pad_token='<PAD>'):
        self.smiles_list = smiles_list
        self.char_to_idx = {**char_to_idx, pad_token: len(char_to_idx), unknown_token: len(char_to_idx) + 1}
        self.max_length = max_length
        self.unknown_idx = self.char_to_idx[unknown_token]
        self.pad_idx = self.char_to_idx[pad_token]

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        idxs = [self.char_to_idx.get(char, self.unknown_idx) for char in smiles]
        idxs = idxs[:self.max_length] + [self.pad_idx] * (self.max_length - len(idxs))
        return torch.tensor(idxs, dtype=torch.long)

