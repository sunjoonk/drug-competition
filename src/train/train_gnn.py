# src/train/train_gnn.py

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from data_utils.dataset import MoleculeDataset
from data_utils.score import set_seed, calculate_score
from models.model_gnn import GINERegressionModel


def run(cfg: dict):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_df = pd.read_csv(os.path.join(cfg["paths"]["data_dir"], "train.csv"))
    test_df = pd.read_csv(os.path.join(cfg["paths"]["data_dir"], "test.csv"))
    sample_submission = pd.read_csv(os.path.join(cfg["paths"]["data_dir"], "sample_submission.csv"))

    # Train/Val split
    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg["training"]["val_ratio"],
        random_state=cfg["seed"]
    )

    train_loader = DataLoader(
        MoleculeDataset(train_df),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        MoleculeDataset(val_df),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False
    )
    test_loader = DataLoader(
        MoleculeDataset(test_df),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False
    )

    # Build model
    sample_batch = next(iter(train_loader))
    model = GINERegressionModel(
        node_input_dim=sample_batch.x.shape[1],
        edge_input_dim=sample_batch.edge_attr.shape[1],
        graph_feat_dim=sample_batch.graph_feat.shape[1],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["model"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10, factor=0.7)

    best_val = 0.0
    patience = 0
    model_dir = cfg["paths"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "gine_final.pt")

    # Training loop
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_feat)
            loss = F.mse_loss(preds, batch.y.view(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_feat)
                y_true.extend(batch.y.view(-1).tolist())
                y_pred.extend(preds.tolist())

        val_score = calculate_score(torch.tensor(y_true), torch.tensor(y_pred))
        scheduler.step(val_score)
        print(f"[Epoch {epoch:03d}] Val Score = {val_score:.4f}")

        if val_score > best_val:
            best_val = val_score
            torch.save(model.state_dict(), model_path)
            patience = 0
        else:
            patience += 1
            if patience >= cfg["training"]["early_stopping"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prediction
    result_dir = cfg["paths"]["result_dir"]
    os.makedirs(result_dir, exist_ok=True)
    sub_path = os.path.join(result_dir, cfg["paths"]["submission_file"])

    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_feat)
            predictions.extend(preds.tolist())

    sample_submission["Inhibition"] = predictions
    sample_submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
