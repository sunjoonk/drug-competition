# Molecular Property Prediction

이 프로젝트는 PyTorch Geometric 기반의 GNN(GINEConv), XGBoost, LightGBM을 활용하여 분자의 SMILES 표현으로부터 `Inhibition` 값을 예측하는 회귀 모델입니다.

## 📁 프로젝트 구조
```
.
├── configs/ # 실험 설정 (YAML)
├── data/ # 학습/예측 데이터 (git에 포함 X)
├── outputs/ # 모델/제출 파일 저장
├── src/
│ ├── main.py # 전체 실행 진입점
│ ├── config_loader.py
│ ├── data_utils/
│ ├── models/
│ ├── train/
│ └── utils/
├── requirements.txt
└── README.md
```

## 🚀 실행 방법

### GNN 학습
```bash
PYTHONPATH=src python src/main.py --config configs/gine.yaml
```

### XGBoost 학습
```bash
PYTHONPATH=src python src/main.py --config configs/xgb_lgb.yaml
```

⚙️ 의존성 설치
```bash
pip install -r requirements.txt
```

🧠 주요 라이브러리
- PyTorch + PyTorch Geometric
- RDKit
- XGBoost / LightGBM
- bayesian-optimization

📄 설정 파일 예시
configs/gine.yaml 또는 configs/xgb_lgb.yaml을 참고하여 하이퍼파라미터를 수정할 수 있습니다.

📦 결과물
- 학습된 모델: outputs/models/*.joblib, *.pt
- 제출 파일: outputs/submissions/*.csv
