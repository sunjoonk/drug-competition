# src/train/train_gnn.py

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from optuna import Trial
import optuna

from data_utils.dataset import MoleculeDataset
from data_utils.score import set_seed, calculate_score
from models.model_gnn import GINERegressionModel


def run(cfg: dict):
    if cfg.get("optuna", {}).get("enabled", False):
        best_params = run_optuna(cfg)
        print("Best hyperparameters from Optuna:", best_params)
    else:
        best_params = cfg["model"]
        best_params["seed"] = cfg["seed"]
        best_params["batch_size"] = cfg["training"]["batch_size"]
        best_params["learning_rate"] = cfg["model"]["learning_rate"]

    final_score = train_and_predict(cfg, best_params)
    print(f"\nFinal Test Score: {final_score:.4f}")


# ------------------------------
# Optuna Objective Function
# ------------------------------
def objective(trial: Trial, cfg: dict, train_df):
    params = {
        "seed": trial.suggest_int("seed", 1, 1000),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 100, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 2, 6),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    }

    set_seed(params["seed"])

    tr_df, vl_df = train_test_split(train_df, test_size=cfg["training"]["val_ratio"], random_state=params["seed"])
    tr_loader = DataLoader(MoleculeDataset(tr_df), batch_size=params["batch_size"], shuffle=True)
    vl_loader = DataLoader(MoleculeDataset(vl_df), batch_size=params["batch_size"], shuffle=False)

    sample_batch = next(iter(tr_loader))
    model = GINERegressionModel(
        node_input_dim=sample_batch.x.shape[1],
        edge_input_dim=sample_batch.edge_attr.shape[1],
        graph_feat_dim=sample_batch.graph_feat.shape[1],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"]
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    device = next(model.parameters()).device

    best_score = 0
    patience = cfg["optuna"].get("patience", 30)
    patience_cnt = 0

    for _ in range(1, 100 + 1):
        model.train()
        for batch in tr_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_feat)
            loss = F.mse_loss(preds, batch.y.view(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in vl_loader:
                batch = batch.to(device)
                preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_feat)
                y_true.extend(batch.y.view(-1).tolist())
                y_pred.extend(preds.tolist())

        val_score = calculate_score(torch.tensor(y_true), torch.tensor(y_pred))
        if val_score > best_score:
            best_score = val_score
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    return best_score


def run_optuna(cfg: dict):
    import pandas as pd
    df = pd.read_csv(os.path.join(cfg["paths"]["data_dir"], "train.csv"))
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, cfg, df),
                   n_trials=cfg["optuna"]["n_trials"],
                   timeout=cfg["optuna"]["timeout"])
    return study.best_params


# ------------------------------
# Final Training + Prediction
# ------------------------------
def train_and_predict(cfg: dict, params: dict):
    import pandas as pd

    set_seed(params["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = cfg["paths"]["data_dir"]
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sample_submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    train_df, val_df = train_test_split(train_df, test_size=cfg["training"]["val_ratio"], random_state=params["seed"])
    train_loader = DataLoader(MoleculeDataset(train_df), batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(MoleculeDataset(val_df), batch_size=params["batch_size"], shuffle=False)
    test_loader = DataLoader(MoleculeDataset(test_df), batch_size=params["batch_size"], shuffle=False)

    sample_batch = next(iter(train_loader))
    model = GINERegressionModel(
        node_input_dim=sample_batch.x.shape[1],
        edge_input_dim=sample_batch.edge_attr.shape[1],
        graph_feat_dim=sample_batch.graph_feat.shape[1],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10, factor=0.7)

    best_val = 0
    cnt = 0
    model_dir = cfg["paths"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "gine_best.pt")

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
            cnt = 0
        else:
            cnt += 1
            if cnt >= cfg["training"]["early_stopping"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final Evaluation
    model.load_state_dict(torch.load(model_path))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_feat)
            y_true.extend(batch.y.view(-1).tolist())
            y_pred.extend(preds.tolist())

    final_score = calculate_score(torch.tensor(y_true), torch.tensor(y_pred))

    # Submission
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

    return final_score
