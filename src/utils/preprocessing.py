import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def smiles_to_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    SMILES 문자열을 RDKit 분자 피처로 변환
    """
    features = []
    for s in df['Canonical_Smiles']:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            features.append([None] * 9)
            continue

        features.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumAromaticRings(mol),
        ])

    feature_df = pd.DataFrame(features, columns=[
        'MolWt', 'LogP', 'HDonors', 'HAcceptors',
        'TPSA', 'RotBonds', 'RingCount', 'HeavyAtoms', 'AromaticRings'
    ])
    return pd.concat([df.drop(columns=['Canonical_Smiles']), feature_df], axis=1)

def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()]
