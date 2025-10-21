# src/data_utils/dataset.py

import pandas as pd
from torch_geometric.data import Dataset
import feature_gnn

class MoleculeDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.has_target = 'Inhibition' in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['Canonical_Smiles']
        inhibition = row['Inhibition'] if self.has_target else None
        return feature_gnn.smile_to_features(smiles, y=inhibition)
