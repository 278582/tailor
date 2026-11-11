# Code release scope

Included: the local Python dependency closure of the selection CLI, pure-Pareto
CLI, metric CLI, LLM/MCTS v2 CLI, Greedy CLI, random MCTS CLI and MIA CLI. Required
package initializers, per-dataset hold-generalization strategies, templates,
context cards and schema metadata are included. Original algorithm implementations,
seeds and feature/distance rules are retained.

Excluded: real/generated tables, models, caches, logs, old run directories,
development tests and validation campaigns, historic batch schedulers, intermediate
code snapshots, backup folders, figures, manuscript files and credentials.

Portability edits are limited to the independent release copy:

- MCTS's fallback reference directory is the local `synthetic/<dataset>`.
- MIA no longer rewrites paths using personal cluster mount aliases.
- The default dataset is the shipped `adult` schema.
- A private LLM proxy and account-specific endpoint are omitted; endpoint and
  credentials are supplied through standard configuration.
- Census natural-frequency geometry is computed from supplied inputs by default;
  reusing geometry from an unpublished historic run is disabled.
- An obsolete pure-Pareto development installation note is removed.
- Query-batch DCR uses the already-existing CPU Manhattan kernel when no CUDA
  callable is bound, preserving the same L1 distances and aggregation protocol.

This package intentionally does not contain original datasets. User-provided data
and any previously found theta are required for real experiments. The README's
small examples do not imply reproduction of full published experiments. The
evaluation protocol must be passed explicitly when comparing methods.
