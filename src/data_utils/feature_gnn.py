# src/data_utils/features_gnn.py
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors


def smile_to_features(smile: str, y: float = None):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None

    # 1. Node features
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append([
            atom.GetAtomicNum(),          # atomic number
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetNumExplicitHs(),
            atom.GetNumImplicitHs(),
            atom.GetMass(),
            atom.GetIsotope(),
            atom.GetNumRadicalElectrons(),
        ])
    x = torch.tensor(node_feats, dtype=torch.float)

    # 2. Edge features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_index += [[i, j], [j, i]]
        edge_feats = [
            bond.GetBondTypeAsDouble(),
            int(bond.IsInRing()),
            int(bond.GetIsConjugated()),
            int(bond.GetStereo()),
        ]
        edge_attr += [edge_feats, edge_feats]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # 3. Graph-level descriptors
    graph_feat = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumAromaticRings(mol),
    ]
    graph_feat = torch.tensor(graph_feat, dtype=torch.float).unsqueeze(0)

    y_tensor = torch.tensor([y], dtype=torch.float) if y is not None else None

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_tensor,
        graph_feat=graph_feat
    )
