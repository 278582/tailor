# tailor
An Efficient Optimization Framework for Enhancing Synthetic Tabular Data Quality via LLM-Guided MCTS

# tailor Main Workflow

This repository contains a tabular synthetic-data post-selection workflow:

- validate synthetic tabular samples against dataset schemas;
- build fidelity, privacy, and utility proxy scores;
- generate random, scalarized, and Pareto post-selection outputs;
- evaluate selected outputs with TabDiff-compatible density, DCR, and utility metrics;
- optionally search theta configurations with LLM/MCTS.

The root environment is intended for the main workflow only:
`post_selection_tool`, `metric_tool`, `llm_mcts_tool`, `postprocess`, and `prompt_pack`.
The `MIA` and `Precison-Recall` folders keep their own specialized instructions.
Training external generators in `third_party` may require their original environments.

## Environment

Create the recommended Conda environment:

```bash
conda env create -f environment.yml
conda activate tailor-main
```

Or use pip in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The environment is CPU-first. For CUDA runs, install a PyTorch build that matches
your local CUDA driver, then reuse the rest of `requirements.txt`.

Quick import check:

```bash
python -c "import numpy,pandas,sklearn,torch,jinja2,sdmetrics,xgboost,prdc,plotly,PIL"
```

## Required Data Layout

The main workflow expects the TabDiff-style data and sample files already present
in this repository:

```text
third_party/TabDiff/data/{dataset}/info.json
third_party/TabDiff/synthetic/{dataset}/real.csv
third_party/TabDiff/synthetic/{dataset}/test.csv
third_party/TabDiff/synthetic/{dataset}/val.csv        # optional
third_party/sample/{source}/{dataset}/sample_*.csv
prompt_pack/dataset_contexts/{dataset}.prompt_context.json
```

Supported v2 datasets are:

```text
adult beijing default diabetes magic news shoppers
```

## Run Post-selection

Small CPU demo on Adult with a pre-generated TabDiff sample:

```bash
python -m post_selection_tool.cli \
  --dataset-name adult \
  --source tabdiff \
  --synthetic-csv third_party/sample/tabdiff/adult/sample_0.csv \
  --exp-name adult_demo \
  --artifact-dir artifacts/postprocess/tabdiff/adult \
  --keep-k 100 \
  --preselect-target 300 \
  --d-cur-size 50 \
  --eval-device cpu \
  --nn-device cpu \
  --disable-progress
```

Core outputs are written under:

```text
artifacts/postprocess/tabdiff/adult/adult_demo/
  input/
  cards/
  validation/
  selection/
  versions/
  report/
```

Important selection CSVs:

```text
versions/selection_random_full.csv
versions/selection_scalar.csv
versions/selection_pareto.csv  # pareto-like version
```

## Evaluate Selections

For a lightweight CPU check, use the PyTorch MLP utility evaluator:

```bash
python -m metric_tool.cli \
  --dataset-name adult \
  --exp-name adult_demo \
  --artifact-dir artifacts/postprocess/tabdiff/adult \
  --eval-device cpu \
  --nn-device cpu \
  --utility-exact-evaluator torch_lightweight_mlp
```

For the TabDiff/XGBoost utility path, use:

```bash
python -m metric_tool.cli \
  --dataset-name adult \
  --exp-name adult_demo \
  --artifact-dir artifacts/postprocess/tabdiff/adult \
  --eval-device auto \
  --nn-device auto \
  --utility-exact-evaluator tabdiff_mle
```

If XGBoost GPU mode is unavailable, the code falls back to CPU histogram mode.
Use `torch_lightweight_mlp` for faster smoke tests.

## Run LLM/MCTS Search

Mock provider smoke run, no API key required:

```bash
python -m llm_mcts_tool.v2_cli \
  --dataset-name adult \
  --artifact-dir artifacts/llm_mcts_v2/adult \
  --exp-name adult_smoke \
  --mode single \
  --single-source tabdiff \
  --provider mock \
  --mcts-budget 1 \
  --theta-proposals-per-event 1 \
  --keep-k 100 \
  --preselect-target 300 \
  --d-cur-size 50 \
  --eval-device cpu \
  --nn-device cpu \
  --utility-exact-evaluator torch_lightweight_mlp \
  --disable-progress
```

LLM-backed run:

```bash
export DASHSCOPE_API_KEY=your_api_key

python -m llm_mcts_tool.v2_cli \
  --dataset-name adult \
  --artifact-dir artifacts/llm_mcts_v2/adult \
  --exp-name adult_llm_mcts_v2 \
  --mode mixed \
  --sources great,smote,tabdiff,tabsyn \
  --provider llm \
  --llm-model qwen3.7-plus \
  --llm-base-url https://dashscope.aliyuncs.com/compatible-mode/v1
```

LLM/MCTS outputs are written under:

```text
artifacts/llm_mcts_v2/{dataset}/{exp_name}/mcts_v2/
  context/
  s_nodes/
  rollouts/
  tree/
  archive/
  final/
```

The final selected theta and table are in:

```text
final/theta_star.json
final/final_pareto.csv  # pareto-like version
```

## Reuse a Found Theta

Run post-selection with a previously found theta:

```bash
python -m post_selection_tool.cli \
  --dataset-name adult \
  --exp-name adult_theta_demo \
  --artifact-dir artifacts/postprocess/tabdiff/adult \
  --theta-mcts-dir artifacts/llm_mcts_v2/adult/adult_smoke/mcts_v2 \
  --theta-source auto \
  --keep-k 100 \
  --preselect-target 300 \
  --d-cur-size 50 \
  --eval-device cpu \
  --nn-device cpu \
  --disable-progress
```

## Tests

Run a small validation set:

```bash
python -m pytest \
  tests/test_validator.py \
  tests/test_metric_reward.py \
  tests/test_pareto_core_selection.py
```

Run the LLM/MCTS contract tests:

```bash
python -m pytest \
  tests/test_llm_mcts_v2_prompt_contract.py \
  tests/test_llm_mcts_v2_source_validation.py \
  tests/test_llm_mcts_v2_uct_depth.py
```

## Notes

- `artifacts/` contains generated outputs and can become large.
- `third_party/TabDiff` is used for dataset metadata and TabDiff-compatible metrics.
- `third_party/sample` contains pre-generated samples used by post-selection and v2 MCTS source-pool selection.
- `MIA/README.md` documents membership-inference audit workflows.
- `Precison-Recall/README.md` documents alpha-Precision and beta-Recall evaluation.
