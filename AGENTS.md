# DKL Tabular Regression Project with Categorical Handling

## 💾 환경 설정

Please generate a Python 3.10+ compatible script that installs the following packages (with compatible versions for CUDA 11.8):

```bash
pip install torch==1.13.1+cu118 torchvision==0.14.1+cu118 torchaudio==0.13.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install gpytorch==1.10.0
pip install scikit-learn pandas matplotlib seaborn
```

---

## ✅ 데이터셋 처리 요구사항

Please create a flexible preprocessor that accepts a Pandas `DataFrame` and does the following:

1. Accepts arguments:
   - `x_columns`: list of feature column names
   - `target_column`: name of regression target column
2. Identifies which of the `x_columns` are categorical (`dtype == 'object' or 'category'`)
3. Performs:
   - **One-hot encoding** for categorical features (using `pd.get_dummies`)
   - **StandardScaler** for numeric features and target
4. Automatically splits the dataset into:
   - 70% train
   - 15% validation
   - 15% test
5. Returns:
   - `X_train, y_train, X_val, y_val, X_test, y_test` (as torch.Tensor)

Save this logic in `data/preprocess.py`

---

## 📁 전체 프로젝트 구성

Generate code for the following structure:

```
dkl_project/
├── data/
│   ├── preprocess.py        # Categorical + numerical preprocessing
│   └── example_dataset.csv  # Sample dataset
├── models/
│   ├── dkl_model.py
│   ├── ensemble_dkl.py
│   └── quantile_dkl.py
├── train/
│   ├── train_base.py
│   ├── train_ensemble.py
│   └── train_quantile.py
├── utils/
│   ├── loss.py
│   ├── metrics.py
│   └── plot.py
├── checkpoints/
├── results/
├── main.py
└── requirements.txt
```

---

## 🧠 모델 요구사항

Implement the following models in PyTorch + GPyTorch:

1. **Base DKL Model**
   - MLP feature extractor
   - GP Layer with 128 inducing points, RBF kernel

2. **Ensemble DKL**
   - 5 independent Base DKL models
   - Aggregate mean and total σ (epistemic + aleatoric)

3. **Quantile Hybrid DKL**
   - Head for 10%, 50%, 90% quantile prediction
   - Loss includes pinball loss for quantiles

All models must support `.to(device)` and fallback to CPU if CUDA is unavailable.

---

## 📉 학습 요구사항

Each of the 3 training scripts must:

- Accept file path to CSV and `x_columns`, `target_column`
- Run preprocessing using `data/preprocess.py`
- Train for max 100 epochs, with early stopping (patience=10)
- Use CosineAnnealingLR scheduler
- Save best model to `checkpoints/`
- Save predictions and plots to `results/`

Each script must save:

- MAE, RMSE, sigma-MAE correlation
- Coverage (% within 95% CI)
- σ vs MAE scatter plot
- Prediction vs True plot
- Quantile Interval plot (quantile model only)

---

## 📦 실행 명령어 예시

Each script must be runnable standalone via CLI:

```bash
python train/train_base.py \
  --csv_path data/example_dataset.csv \
  --x_columns "feature1,feature2,cat_feature1,cat_feature2" \
  --target_column "target"

python train/train_ensemble.py ...  # 동일하게 동작

python train/train_quantile.py ...  # 동일하게 동작
```

Use `argparse` to receive arguments.

---

## 🧾 requirements.txt

Please generate a requirements.txt with all versions explicitly pinned for full reproducibility.

---

## 📘 참고 데이터

If `data/example_dataset.csv` does not exist, synthesize a mixed-type dataset using `sklearn.datasets.make_regression` and append categorical columns using:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=2000, n_features=5, noise=0.1, random_state=42)
df = pd.DataFrame(X, columns=[f'num_{i}' for i in range(5)])
df['cat_feature1'] = np.random.choice(['A', 'B', 'C'], size=2000)
df['cat_feature2'] = np.random.choice(['X', 'Y'], size=2000)
df['target'] = y
df.to_csv('data/example_dataset.csv', index=False)
```

---

## 📦 산출물

- Save Python scripts in respective folders above
- Output results in `/results/` with timestamps or model names
- Save best models in `/checkpoints/`
