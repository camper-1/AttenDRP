import numpy as np
import torch
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from early_stopping import EarlyStopping




def train_model(model, train_loader, val_loader, device, optimizer, scheduler, criterion, num_epochs=100, patience=10,
                label='', save_features=False, feature_save_path="./features/", drug_list=None, cell_line_list=None):
    """
    训练模型并支持保存中间特征表示，同时保存药物和细胞系索引
    """
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    metrics = {"train_rmse": [], "val_rmse": [], "pcc": [], "r2": []}

    # 创建特征保存目录
    if save_features:
        import os
        os.makedirs(feature_save_path, exist_ok=True)

    # 保存药物和细胞系的索引
    if save_features and drug_list is not None and cell_line_list is not None:
        np.save(f"{feature_save_path}/drug_indices.npy", np.array(drug_list))
        np.save(f"{feature_save_path}/cell_indices.npy", np.array(cell_line_list))
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
        y_pred_list = []
        y_true_list = []

        # 用于存储中间特征
        all_drug_features = []
        all_cell_features = []

        with torch.no_grad():
            for drug_data, cell_data, target in val_loader:
                drug_data, cell_data, target = drug_data.to(device), cell_data.to(device), target.to(device)
                outputs = model(drug_data, cell_data)
                val_loss += criterion(outputs, target).item()

                y_pred_list.append(outputs.cpu().numpy())
                y_true_list.append(target.cpu().numpy())

                # 提取中间特征（假设模型具有 `forward_features` 方法）
                if save_features:
                    drug_features, cell_features = model.forward_features(drug_data, cell_data)
                    all_drug_features.append(drug_features.cpu().numpy())
                    all_cell_features.append(cell_features.cpu().numpy())

        val_loss /= len(val_loader)
        y_pred_list = np.concatenate(y_pred_list)
        y_true_list = np.concatenate(y_true_list)

        # 计算 RMSE, PCC 和 R²
        val_rmse = np.sqrt(val_loss)
        pcc, _ = pearsonr(y_pred_list.flatten(), y_true_list.flatten())
        r2 = r2_score(y_true_list, y_pred_list)

        print(f"[{label}] Epoch {epoch + 1}/{num_epochs}, Val RMSE: {val_rmse}, PCC: {pcc}, R²: {r2}")

        # 记录每个 epoch 的指标
        metrics["train_rmse"].append(np.sqrt(train_loss))
        metrics["val_rmse"].append(val_rmse)
        metrics["pcc"].append(pcc)
        metrics["r2"].append(r2)

        if scheduler is not None:
            scheduler.step(val_rmse)

        # 保存特征表示
        if save_features:
            drug_features_path = f"{feature_save_path}/drug_features_epoch_{epoch + 1}.npy"
            cell_features_path = f"{feature_save_path}/cell_features_epoch_{epoch + 1}.npy"
            np.save(drug_features_path, np.concatenate(all_drug_features, axis=0))
            np.save(cell_features_path, np.concatenate(all_cell_features, axis=0))
            print(f"Features saved: {drug_features_path}, {cell_features_path}")

        # 早停
        early_stopping(val_rmse, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 加载最佳模型权重
    model.load_state_dict(torch.load('checkpoint.pt', map_location=device))

    # 保存最终模型
    torch.save(model.state_dict(), './data/processed/trained_model.pth')

    return metrics

