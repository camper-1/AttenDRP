import os
import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from rdkit import Chem
from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

seed_value = 1
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt_ge=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, saliency_map=False, test_drug_dict=None):

        # root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.saliency_map = saliency_map
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt_ge, y, smile_graph, test_drug_dict)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # brief Customize the process method to fit the task of drug-cell line ic50 prediction
    # Inputs: param XD - drug SMILES, XT: cell line features, param Y: list of labels (i.e. ic50)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt_ge, y, smile_graph, test_drug_dict):
        assert (len(xd) == len(xt_ge) and len(xt_ge) == len(y)), "The four lists must be the same length!"
        data_list = []
        data_len = len(xd)
        print(data_len)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))
            smiles = xd[i]
            target_ge = xt_ge[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))

            # require_grad of cell-line for saliency map
            if self.saliency_map == True:
                GCNData.target_ge = torch.tensor([target_ge], dtype=torch.float, requires_grad=True)
            else:
                GCNData.target_ge = torch.FloatTensor([target_ge])

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

    def getXD(self):
        return self.xd

def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse

def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse

def mse_cust(y, f):
    mse = ((y - f) ** 2)
    return mse

def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp

def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci

def draw_loss(train_losses, test_losses, title):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    plt.savefig(title + ".png")  # should before show method

def draw_pearson(pearsons, title):
    plt.figure()
    plt.plot(pearsons, label='test pearson')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(title + ".png")  # should before show method

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data():
    drug_smiles_file = 'data/processed/drug_features.npy'
    cell_data_file = 'data/processed/cell_data.npy'
    labels_file = 'data/labels.npy'

    # 检查文件是否存在
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
    print(f"Data saved to {file_path}")

def is_valid_smiles(smiles):
    """
    检查 SMILES 字符串是否有效
    :param smiles: SMILES 字符串
    :return: True 如果 SMILES 有效，否则为 False
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def create_graph_data(topo_features, edge_indices=None):
    """
    创建图数据对象
    :param topo_features: 节点的特征矩阵 [num_nodes, feature_dim]
    :param edge_indices: 表示节点间连接关系的边索引 [2, num_edges]，如果没有提供，假设邻接关系
    :return: PyG 的 Data 对象
    """
    # 首先将 topo_features 转换为张量
    topo_features = torch.tensor(topo_features, dtype=torch.float)

    # 检查 topo_features 的维度是否为 2D，如果是 1D，则通过 unsqueeze 扩展为 2D
    if len(topo_features.shape) == 1:
        topo_features = topo_features.unsqueeze(1)  # 扩展为 [num_nodes, 1] 形状

    # 如果特征是 1 维（[num_nodes, 1]），可以扩展到 [num_nodes, 10] 或 [num_nodes, feature_dim]
    if topo_features.shape[1] == 1:
        topo_features = topo_features.repeat(1, 10)  # 扩展到 10 维特征

    # 如果没有提供 edge_index，使用简单的环形结构
    num_nodes = topo_features.shape[0]
    if edge_indices is None:
        edge_index = torch.tensor([[i, (i+1) % num_nodes] for i in range(num_nodes)], dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long)

    # 创建 PyG Data 对象
    data = Data(x=topo_features, edge_index=edge_index)
    return data

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")










