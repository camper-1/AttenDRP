import numpy as np
import torch
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from early_stopping import EarlyStopping
import os

def train_model(model, train_loader, val_loader, device, optimizer, scheduler, criterion,
                num_epochs=100, patience=10, label='', save_features=False,
                feature_save_path="./features/", drug_list=None, cell_line_list=None):

    os.makedirs(feature_save_path, exist_ok=True)

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=os.path.join(feature_save_path, 'checkpoint.pt'))
    metrics = {"train_rmse": [], "val_rmse": [], "pcc": [], "r2": []}

    if save_features and drug_list is not None and cell_line_list is not None:
        np.save(os.path.join(feature_save_path, "drug_indices.npy"), np.array(drug_list))
        np.save(os.path.join(feature_save_path, "cell_indices.npy"), np.array(cell_line_list))
        print("Drug and cell line indices saved.")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for drug_data, cell_data, target in train_loader:
            drug_data, cell_data, target = drug_data.to(device), cell_data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(drug_data, cell_data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        y_pred_list, y_true_list = [], []
        all_drug_features, all_cell_features = [], []

        with torch.no_grad():
            for drug_data, cell_data, target in val_loader:
                drug_data, cell_data, target = drug_data.to(device), cell_data.to(device), target.to(device)
                outputs = model(drug_data, cell_data)
                val_loss += criterion(outputs, target).item()
                y_pred_list.append(outputs.cpu().numpy())
                y_true_list.append(target.cpu().numpy())

                if save_features:
                    drug_f, cell_f = model.forward_features(drug_data, cell_data)
                    all_drug_features.append(drug_f.cpu().numpy())
                    all_cell_features.append(cell_f.cpu().numpy())

        val_loss /= len(val_loader)
        y_pred_list = np.concatenate(y_pred_list)
        y_true_list = np.concatenate(y_true_list)

        val_rmse = np.sqrt(val_loss)
        pcc, _ = pearsonr(y_pred_list.flatten(), y_true_list.flatten())
        r2 = r2_score(y_true_list, y_pred_list)

        print(f"[{label}] Epoch {epoch+1}/{num_epochs}, Train RMSE: {np.sqrt(train_loss):.4f}, "
              f"Val RMSE: {val_rmse:.4f}, PCC: {pcc:.4f}, R²: {r2:.4f}")

        metrics["train_rmse"].append(np.sqrt(train_loss))
        metrics["val_rmse"].append(val_rmse)
        metrics["pcc"].append(pcc)
        metrics["r2"].append(r2)

        if scheduler is not None:
            scheduler.step(val_rmse)

        if save_features:
            np.save(os.path.join(feature_save_path, f"drug_features_epoch_{epoch+1}.npy"),
                    np.concatenate(all_drug_features, axis=0))
            np.save(os.path.join(feature_save_path, f"cell_features_epoch_{epoch+1}.npy"),
                    np.concatenate(all_cell_features, axis=0))

        early_stopping(val_rmse, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(os.path.join(feature_save_path, 'checkpoint.pt'), map_location=device))
    torch.save(model.state_dict(), os.path.join(feature_save_path, 'trained_model.pth'))

    return metrics

