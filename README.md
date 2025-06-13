# DKL GPR Comparison Project

This repository demonstrates Deep Kernel Learning (DKL) models for tabular regression with categorical handling. The project structure and scripts are generated according to `AGENTS.md` instructions.

Run the base model training as:

```bash
python dkl_project/train/train_base.py \
  --csv_path dkl_project/data/example_dataset.csv \
  --x_columns "num_0,num_1,num_2,num_3,num_4,cat_feature1,cat_feature2" \
  --target_column "target"
```

Other modes (`ensemble` and `quantile`) can be invoked with their respective scripts under `dkl_project/train/`.
