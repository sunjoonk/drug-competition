# src/train/train_xgb.py

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

from utils.evaluate import calculate_score
from utils.preprocessing import smiles_to_data, remove_duplicate_columns
from utils.modeling import PrunableBayesianOptimizer


def run(cfg: dict):
    SEED = cfg["seed"]
    np.random.seed(SEED)

    # Load data
    data_dir = cfg["paths"]["data_dir"]
    model_dir = cfg["paths"]["model_dir"]
    result_dir = cfg["paths"]["result_dir"]
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    submission_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    train_df = remove_duplicate_columns(smiles_to_data(train_df))
    test_df = remove_duplicate_columns(smiles_to_data(test_df))

    x = train_df.drop(columns=["Inhibition"])
    y = train_df["Inhibition"]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=cfg["training"]["val_ratio"], random_state=SEED)

    def xgb_black(**kwargs):
        int_keys = ["n_estimators", "max_depth", "min_child_weight", "max_delta_step"]
        params = {
            k: int(round(v)) if k in int_keys else v
            for k, v in kwargs.items()
        }
        params.update({
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": SEED,
            "eval_metric": "rmse"
        })

        es = EarlyStopping(rounds=50, metric_name="rmse", data_name="validation_0", save_best=True)
        model = XGBRegressor(**params, callbacks=[es])
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0)
        pred = model.predict(x_val)
        return -np.sqrt(mean_squared_error(y_val, pred))

    optimizer = PrunableBayesianOptimizer(
        f=xgb_black,
        pbounds=cfg["xgb_params"],
        random_state=SEED,
        patience=60
    )

    print("[XGBoost] Searching best hyperparameters...")
    start = time.time()
    optimizer.maximize(init_points=20, n_iter=100)
    print(f"Time taken: {time.time() - start:.2f}s")

    best_params = optimizer.max["params"]
    for k in ["n_estimators", "max_depth", "min_child_weight", "max_delta_step"]:
        best_params[k] = int(round(best_params[k]))

    best_params.update({
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": SEED,
        "eval_metric": "rmse"
    })

    # Train final model
    model = XGBRegressor(**best_params, callbacks=[
        EarlyStopping(rounds=50, metric_name="rmse", data_name="validation_0", save_best=True)
    ])
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0)
    joblib.dump(model, os.path.join(model_dir, "xgb_regressor.joblib"))

    # Predict + evaluate
    pred = model.predict(x_val)
    score = calculate_score(y_val, pred)
    print(f"[XGBoost] Validation Score: {score:.4f}")

    # Submit
    submission_df["Inhibition"] = model.predict(test_df)
    submission_df.to_csv(os.path.join(result_dir, cfg["paths"]["submission_file_xgb"]), index=False)
    print(f"Submission saved to {cfg['paths']['submission_file_xgb']}")
