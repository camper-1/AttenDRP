import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# 分子指纹特征：根据输入 SMILES 生成分子指纹
def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    """
    将 SMILES 转换为分子指纹特征
    :param smiles: SMILES 字符串
    :param radius: Morgan 指纹半径
    :param nBits: 指纹比特位数
    :return: 分子指纹特征向量
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # 生成 Morgan 指纹，如果遇到警告可以忽略
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))
    else:
        return np.zeros(nBits)  # 若无效 SMILES 返回全零向量

# 拓扑特征
def smiles_to_topological_features(smiles):
    """
    将 SMILES 转换为拓扑结构特征（例如重原子计数、环数量等）
    :param smiles: SMILES 字符串
    :return: 拓扑特征向量
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * 10  # 如果分子无效，返回全零的特征

    heavy_atoms = Descriptors.HeavyAtomCount(mol)
    rings = Descriptors.RingCount(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    mol_wt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    mol_wt_per_heavy_atom = mol_wt / heavy_atoms if heavy_atoms != 0 else 0

    topo_features = [
        heavy_atoms, rings, rotatable_bonds, aromatic_rings,
        h_donors, h_acceptors, tpsa, mol_wt, logp, mol_wt_per_heavy_atom
    ]
    return topo_features

# 理化性质特征
def smiles_to_physicochemical_properties(smiles):
    """
    将 SMILES 转换为理化性质特征
    :param smiles: SMILES 字符串
    :return: 理化性质特征向量
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array([
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.MolMR(mol),
            Descriptors.MolWt(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol)
        ])
    else:
        return np.zeros(9)  # 扩展为9维特征
