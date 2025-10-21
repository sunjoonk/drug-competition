# src/main.py

import argparse
from config_loader import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/gine.yaml",
        help="Path to config YAML file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_type = cfg.get("model_type", "gine").lower()

    if model_type == "gine":
        from train.train_gnn import run as run_gnn
        run_gnn(cfg)

    elif model_type == "xgboost":
        from train.train_xgb import run as run_xgb
        run_xgb(cfg)

    elif model_type == "lgbm":
        from train.train_lgbm import run as run_lgbm
        run_lgbm(cfg)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


if __name__ == "__main__":
    main()
