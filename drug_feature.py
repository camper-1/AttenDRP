import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits), dtype=np.float32)
    else:
        return np.zeros(nBits, dtype=np.float32)

def smiles_to_topological_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(10, dtype=np.float32)

    heavy_atoms = Descriptors.HeavyAtomCount(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    mol_wt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    mol_wt_per_heavy_atom = mol_wt / heavy_atoms if heavy_atoms != 0 else 0.0

    return np.array([
        heavy_atoms, rings, rotatable_bonds, aromatic_rings,
        h_donors, h_acceptors, tpsa, mol_wt, logp, mol_wt_per_heavy_atom
    ], dtype=np.float32)

def smiles_to_physicochemical_properties(smiles):
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
        ], dtype=np.float32)
    else:
        return np.zeros(9, dtype=np.float32)

