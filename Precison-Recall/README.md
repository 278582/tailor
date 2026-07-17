# Alpha-Precision / Beta-Recall Evaluation

This directory contains a standalone evaluator for tabular synthetic data. It
reads real tables from `third_party/TabDiff/synthetic/{dataset}/real.csv` and
synthetic tables from:

```text
artifacts/postprocess/{model}/{dataset}/{exp}/versions/{data}.csv
```

The evaluator uses TabDiff metadata in `third_party/TabDiff/data/{dataset}/info.json`
to split numeric, categorical, and target columns. By default, target columns are
included so the metric evaluates the full joint table distribution.

Categorical values are normalized with `str.strip()` inside the evaluator. This
does not modify source CSV files, but prevents values such as `" Male"` and
`"Male"` from being treated as different categories.

## Install

```bash
/mnt/lustre/liuzhiwei/miniconda3/bin/conda create -n tgm_pr python=3.10 -y
/mnt/lustre/liuzhiwei/miniconda3/bin/conda install -n tgm_pr -c conda-forge numpy pandas scipy scikit-learn tqdm -y
```

## Validate Seven Tabular Datasets

```bash
/mnt/lustre/liuzhiwei/miniconda3/bin/conda run -n tgm_pr python Precison-Recall/alpha_beta_pr.py \
  --datasets shoppers news adult default diabetes magic beijing \
  --model tabdiff \
  --exp no_theta_0_1 \
  --data selection_random_full \
  --max-rows 20000 \
  --num-points 101 \
  --seed 42 \
  --out-dir Precison-Recall/results
```

## Evaluate Another Synthetic Output

Change `--model`, `--exp`, and `--data`:

```bash
/mnt/lustre/liuzhiwei/miniconda3/bin/conda run -n tgm_pr python Precison-Recall/alpha_beta_pr.py \
  --datasets shoppers news adult default diabetes magic beijing \
  --model MODEL \
  --exp EXP \
  --data DATA \
  --max-rows 20000 \
  --num-points 101 \
  --seed 42 \
  --out-dir Precison-Recall/results
```

Outputs:

- `summary.csv`: one row per dataset with scalar scores.
- `curves.json`: full alpha-Precision and beta-Recall curves.

Important scalar fields:

- `alpha_precision_at_0.95`: alpha-Precision at alpha=0.95.
- `beta_recall_at_0.95`: backward-compatible alias of the SynthCity-style
  beta coverage score.
- `beta_coverage_at_0.95`: same value as `beta_recall_at_0.95`, named to make
  the coverage semantics explicit.
- `beta_support_recall_at_0.95`: support-style recall based on distance to the
  synthetic distribution center.
- `authenticity_fixed`: corrected authenticity calculation.
- `authenticity_legacy`: previous compatibility calculation.
- `categorical_exact_overlap_min` and `categorical_strip_overlap_min`: category
  overlap diagnostics before and after whitespace normalization.
