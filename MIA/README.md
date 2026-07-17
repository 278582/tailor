# MIA: Membership Inference Audit for Tabular Synthetic Data

This folder contains a standalone privacy audit tool for released tabular synthetic data.
It reads the repository's postprocess artifacts directly and reports membership inference
risk for each synthetic selection.

## Threat Models

- `release_only`: attacks use only the released synthetic table plus target records.
- `release_only_calibrated`: attacks additionally use a reference/control split for calibration.
- `upper_bound_not_release_only`: cross-validated classifier over attack scores; useful as an upper-bound diagnostic.
- `shadow_attack`: enabled only when one or more `--shadow-run-dir` directories are supplied.

## Built-in Attacks

- `exact_match`: flags target records that appear exactly in synthetic data.
- `nearest_neighbor`: scores records by negative nearest-neighbor distance to synthetic data.
- `density_ratio`: DOMIAS-inspired kNN density ratio against a reference split.
- `attribute_error`: MIA-EPT-inspired per-column prediction error profile.
- `supervised_error_profile`: optional upper-bound classifier over the release-only attack scores.
- `shadow_attack`: optional classifier trained from supplied shadow runs with matching selection names.

## Environment

```bash
/mnt/lustre/liuzhiwei/miniconda3/bin/conda create -n MIA --override-channels -c conda-forge python=3.10 -y
/mnt/lustre/liuzhiwei/miniconda3/bin/conda install -n MIA --override-channels -c conda-forge \
  numpy pandas scipy scikit-learn joblib tqdm xgboost pytest matplotlib seaborn -y
```

`anonymeter` is documented as an optional comparator, but this implementation does not
require it. Install it separately only when the current pip index is reachable.

## Usage

Audit all selections in a postprocess run:

```bash
/mnt/lustre/liuzhiwei/miniconda3/bin/conda run -n MIA \
  python -m MIA.cli \
  --run-dir artifacts/postprocess/tabsyn/news/no_theta_1_2 \
  --all-selections \
  --out-dir MIA/results/news_no_theta_1_2
```

For quick scans on large tables, add row caps such as:

```bash
python -m MIA.cli \
  --run-dir artifacts/postprocess/tabsyn/news/no_theta_1_2 \
  --selection-name pareto \
  --max-member-rows 5000 \
  --max-nonmember-rows 5000 \
  --max-synthetic-rows 5000 \
  --out-dir MIA/results/news_quick_pareto
```

Audit one explicit synthetic CSV:

```bash
python -m MIA.cli \
  --train-csv path/to/eval_train.csv \
  --control-csv path/to/eval_holdout.csv \
  --reference-csv path/to/eval_test.csv \
  --synthetic-csv path/to/selection_pareto.csv \
  --out-dir MIA/results/example
```

Add shadow runs when available:

```bash
python -m MIA.cli \
  --run-dir artifacts/postprocess/tabsyn/news/no_theta_1_2 \
  --selection-name pareto \
  --shadow-run-dir artifacts/postprocess/tabsyn/news/no_theta_0_1 \
  --shadow-run-dir artifacts/postprocess/tabsyn/news/no_theta_3_1 \
  --out-dir MIA/results/news_shadow_pareto
```

## Outputs

- `summary.json`: metrics and best attack per selection.
- `metrics.csv`: flat comparison table across selections.
- `<selection>/scores.csv`: per-row labels and attack scores.
- `<selection>/attack_details.json`: method metadata and diagnostics.
