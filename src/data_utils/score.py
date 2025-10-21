# src/data_utils/score.py
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    RMSE & Pearson 기반 맞춤 스코어 계산 (클립 제한 포함)
    """
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    denominator = torch.max(y_true) - torch.min(y_true)
    A = rmse / denominator if denominator != 0 else torch.tensor(0.0, device=y_true.device)

    y_mean = torch.mean(y_true)
    y_pred_mean = torch.mean(y_pred)
    cov = torch.mean((y_true - y_mean) * (y_pred - y_pred_mean))
    std_y = torch.std(y_true)
    std_y_hat = torch.std(y_pred)

    if std_y * std_y_hat == 0:
        B = torch.tensor(0.0, device=y_true.device)
    else:
        B = torch.clamp(cov / (std_y * std_y_hat), 0.0, 1.0)

    score = 0.5 * (1 - torch.minimum(A, torch.tensor(1.0, device=y_true.device))) + 0.5 * B
    return score.item()
