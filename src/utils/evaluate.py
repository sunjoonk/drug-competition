import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def calculate_score(y_true, y_pred) -> float:
    """
    커스텀 평가 지표:
    - 정규화된 RMSE + PCC 기반 혼합 점수
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    A = rmse / (y_true.max() - y_true.min()) if y_true.max() != y_true.min() else 0.0

    try:
        pcc, _ = pearsonr(y_true, y_pred)
        pcc = np.clip(pcc, 0.0, 1.0)
    except:
        pcc = 0.0

    score = 0.5 * (1 - min(A, 1.0)) + 0.5 * pcc
    return score
