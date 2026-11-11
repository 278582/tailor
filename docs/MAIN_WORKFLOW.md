# Main workflow and references

## Selection

`post_selection_tool.cli` loads the supplied real/synthetic data, validates
candidate rows, constructs metadata/cards, reserves the current-distribution
reference, preselects candidates and computes fidelity, privacy and utility
objectives. The generic route constructs random, scalar, preselected, floor and
Pareto selections, followed by configured repair/reward stages.

The `hold_generalization` route wraps the same anchor preparation and dispatches
to the dataset-specific production recipes. It requires a supplied fixed
reference directory and a supplied sample bank. For the seven smaller datasets,
the command structure is:

```bash
python -m post_selection_tool.cli \
  --dataset-name adult --source tabdiff --exp-name final_0 \
  --synthetic-csv third_party/sample-2/tabdiff/adult/sample_0.csv \
  --selection-pipeline hold_generalization \
  --reference-split-dir synthetic/adult \
  --holdgen-runtime-dir outputs/adult/tabdiff/final_0_runtime \
  --holdgen-pool-csvs third_party/sample-2/tabdiff/adult/sample_0.csv third_party/sample-2/tabdiff/adult/sample_1.csv \
  --holdgen-sample-id 0 --seed 20260902 \
  --artifact-dir outputs/adult/tabdiff \
  --keep-k 29274 --preselect-target 40984 --d-cur-size 2000 \
  --density-reference-size 5000 --privacy-version v2 \
  --dcr-signal-full-reference --theta-col-ps-all-columns \
  --nn-device cuda:0 --eval-device cuda:0 \
  --disable-progress --skip-validation-records
```

These keep/preselect values are dataset-specific; ensure your supplied files
have enough legal candidates. A chosen strategy can be loaded with the existing
`--theta-mcts-dir`/`--theta-source` arguments. No theta file is bundled or silently
fabricated. Inspect the CLI for quality, swap-budget and reward options rather
than treating this structural example as a tuned configuration.

For US Census 1990, use `--selection-pipeline hold_generalization` with
`--census-options-json configs/census_natural_frequency_options.json` or the
deduplicated variant `configs/census_hold_generalization_options.json`. The
natural-frequency configuration preserves distinct source-record identities even
when values are equal. Its default recomputes distances rather than relying on
external historical caches. Both Census routes retain staged fingerprint checks
and require a fresh output/runtime directory unless explicitly resuming supported
completed stages.

## Evaluation

`metric_tool.cli` requires the selected CSV and the originating experiment's
cards. It computes shape/trend against real train, utility on the final test and
DCR against train/test. `--enable-hold-dcr` adds the train/hold result.

Always state the DCR protocol:

| Mode | Actual reference/query behavior |
| --- | --- |
| `paired_nn_coreset` (generic default) | Equal sampled syn/reference counts; complete train NN; strict `<` Share |
| `query_batches_full_train_v1` | Complete train and reference pools; syn-only shuffled full batches; strict `<` Share; incomplete query remainder excluded from the primary result |

For the query-batch mode use equal hold/test row counts, `--enable-hold-dcr`,
`--dcr-cap 0` and explicitly choose 1-4 rounds (the main comparisons used 2 and seed
20261027). Distances use the existing train-range-scaled numeric and one-hot L1
space, including the target. A Share target near 0.5 is not a universal finite
sample guarantee or a privacy-loss probability. The row-level repair signal in
the legacy route is not automatically the query-batch evaluation protocol.

Outputs include per-version `metrics_summary.json`, component details, utility
reports, DCR distances and configured repeat/batch plans. `tabdiff_mle` selects the
TabDiff-compatible XGBoost utility path; `torch_lightweight_mlp` is the alternative
lightweight evaluator.

## Searches and privacy audits

LLM/MCTS, random MCTS and Greedy propose/select column policies; the rollout uses
the selection/evaluation adapters. Search feedback can include test DCR/utility,
so do not label those outer searches test-independent. A fixed theta may be reused
on additional synthetic sample CSVs with explicit provenance.

MIA uses supplied train members, hold nonmembers and a test density reference.
Use the explicit CSV CLI for unambiguous paths. The five core attacks are exact
match, nearest neighbour, density ratio, attribute error and the optional
supervised error profile. Keep all attack results; any post-hoc subset analysis
must be separately labelled. No historical summaries or favourable example
subsets are part of this code release.
