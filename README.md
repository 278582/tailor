# TGM: tabular synthetic-data post-selection

This code release contains the production selection, evaluation and column-policy
search workflows. Real datasets, generated samples, trained models, previous
results, development experiments and API credentials are supplied separately.

## Installation

Python 3.10 or newer and Linux are recommended. Some runtime modules use POSIX
file locks. Run commands from this repository directory.

```bash
conda env create -f environment.yml
conda activate tgm-main
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA driver for GPU execution. The original
high-cardinality/Diabetes compact-distance implementations use `cuda:0` directly;
these dataset routes require CUDA. Small legacy selections, the ordinary DCR
distance kernel, and MIA can run on CPU. Exact TabDiff utility uses XGBoost and
can take substantially longer than the lightweight PyTorch MLP evaluator.

Optional Ark provider: `pip install -r requirements-llm-optional.txt`.
No external LLM request is made by installing or importing this code.

## Modules

| Module | Main entrypoint | Purpose |
| --- | --- | --- |
| `post_selection_tool` | `python -m post_selection_tool.cli` | Selection, floor, repair and reward; legacy and hold-generalization routes |
| Pure Pareto | `python -m post_selection_tool.init_pareto_cli` | Exact Pareto baseline without floor/repair/strategy search |
| `metric_tool` | `python -m metric_tool.cli` | Shape, trend, DCR, utility and reward |
| `llm_mcts_tool` | `python -m llm_mcts_tool.v2_cli` | LLM-guided strategy/MCTS search; a mock provider is available |
| `greedy` | `python -m greedy` | Random-edit greedy column-policy baseline |
| `mcts` | `python -m mcts` | Random-edit MCTS baseline, no LLM |
| `MIA` | `python -m MIA.cli` | Membership-inference audits |

`postprocess/`, prompt templates and dataset context cards are supporting runtime
dependencies. Only the necessary TabDiff evaluation files and schema metadata are
included in `third_party/TabDiff/`; generator training is not included.

## Provide data

```text
synthetic/<dataset>/train.csv
synthetic/<dataset>/hold.csv
synthetic/<dataset>/test.csv
third_party/sample-2/<model>/<dataset>/sample_0.csv
third_party/sample-2/<model>/<dataset>/sample_1.csv
```

CSV headers, column types and target must match
`third_party/TabDiff/data/<dataset>/info.json`. The shipped schemas are Adult,
Beijing, Default, Diabetes, Magic, News, Shoppers and US Census 1990. Metadata and
prompt row counts may describe the upstream datasets; runtime CSVs determine the
actual counts. Schema/type and target definitions are unchanged in this release.

Data preparation and splitting are deliberately outside this code package. Use
your already-fixed train/hold/test, including `|hold| = |test|` for paired reporting
with `query_batches_full_train_v1`. `--reference-split-dir` names a directory
containing `hold.csv` and `test.csv`; train still comes from
`synthetic/<dataset>/train.csv`. This argument does not split or overwrite data.

## Run selection and evaluation

The following small-budget example uses user-supplied Adult files and exercises
the generic legacy route. Use suitable `keep-k`, preselection and dataset options
for your actual experiment. See `docs/MAIN_WORKFLOW.md` for the hold-generalization
route and the separation of DCR evaluation modes.

```bash
python -m post_selection_tool.cli \
  --dataset-name adult --source tabdiff \
  --synthetic-csv third_party/sample-2/tabdiff/adult/sample_0.csv \
  --reference-split-dir synthetic/adult \
  --artifact-dir outputs/adult/tabdiff --exp-name demo \
  --keep-k 100 --preselect-target 140 --d-cur-size 20 \
  --density-reference-size 5000 \
  --privacy-version v2 --nn-device cuda:0 --eval-device cuda:0 \
  --disable-progress --skip-validation-records
```

```bash
python -m metric_tool.cli \
  --dataset-name adult --artifact-dir outputs/adult/tabdiff --exp-name demo \
  --reference-split-dir synthetic/adult \
  --selection-csv outputs/adult/tabdiff/demo/versions/selection_pareto.csv \
  --selection-name tailor --eval-dir outputs/adult/tabdiff/demo/eval \
  --metrics-output outputs/adult/tabdiff/demo/report/tailor_metrics_summary.json \
  --seed 20260915 --density-reference-size 5000 --privacy-version v2 \
  --utility-exact-evaluator tabdiff_mle \
  --dcr-align-mode query_batches_full_train_v1 \
  --dcr-repeats 2 --dcr-seed 20261027 --dcr-cap 0 --enable-hold-dcr \
  --nn-device cuda:0 --eval-device cuda:0
```

The selected table must have at least as many rows as the hold/test batch size
for this DCR mode. Raise the example's `keep-k` accordingly, or use
`paired_nn_coreset` for smaller queries. Selectors write their runtime cards along
with outputs; `metric_tool` needs those cards, not just a bare selected CSV.

## Strategy search

```bash
python -m llm_mcts_tool.v2_cli \
  --dataset-name adult --artifact-dir outputs/llm_mcts/adult \
  --exp-name mock_demo --mode single --single-source tabdiff \
  --provider mock --sample-root third_party/sample-2 \
  --mcts-budget 1 --theta-proposals-per-event 1 \
  --keep-k 100 --preselect-target 140 --d-cur-size 20 \
  --utility-exact-evaluator torch_lightweight_mlp \
  --nn-device cpu --eval-device cpu --disable-progress
```

For real LLM use, configure provider credentials in your environment or in a local
`.env` that is not committed. `.env.example` contains names and empty values only.
Choose the model and endpoint explicitly with `--llm-model` and `--llm-base-url`.
Random Greedy/MCTS use their `--sample-root`, `--reference-split-dir`,
`--dcr-align-mode` and other CLI options; inspect `--help` before running.

## MIA

```bash
python -m MIA.cli \
  --train-csv synthetic/adult/train.csv \
  --control-csv synthetic/adult/test.csv \
  --reference-csv synthetic/adult/hold.csv \
  --synthetic-csv outputs/adult/tabdiff/demo/versions/selection_pareto.csv \
  --selection-name tailor --out-dir outputs/mia/adult/tailor \
  --seed 20260420 --density-k 5 --max-attribute-columns 20
```

All row caps default to zero (full data). Train is the member pool; hold is the
nonmember/control pool; test calibrates the density attack. When hold was used
during selection, document that provenance. The supervised error-profile attack
is an upper-bound diagnostic, not a release-only attack.

## Release contents

See `docs/RELEASE_SCOPE.md` and `CODE_MANIFEST.json`. The release retains actual
transitive dependencies: modules with old version numbers can still implement
the current repair call chain. They are not removable merely by filename.
Historical launchers, validation experiments and result aggregation scripts are
not included. Original production files were not changed while assembling this
release. `THIRD_PARTY_NOTICES.md` describes the bundled TabDiff code and its license.
