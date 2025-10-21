# src/config_loader.py
import yaml
from pathlib import Path

def load_config(path: str = "configs/gine.yaml"):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
