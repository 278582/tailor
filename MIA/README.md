# Membership-inference audit

The standalone entrypoint is `python -m MIA.cli`. It accepts explicit CSV paths or
a TGM experiment directory. Install the root requirements or this module's
requirements. See the root README for the explicit-path command.

```bash
python -m MIA.cli \
  --train-csv synthetic/adult/train.csv \
  --control-csv synthetic/adult/hold.csv \
  --reference-csv synthetic/adult/test.csv \
  --synthetic-csv path/to/released_table.csv \
  --selection-name tailor --out-dir outputs/mia/adult/tailor \
  --seed 20260420 --density-k 5 --max-attribute-columns 20
```

Rows are uncapped by default. `--max-member-rows`, `--max-nonmember-rows`,
`--max-synthetic-rows` and `--max-reference-rows` are explicit optional caps.
`--no-supervised-profile` disables the upper-bound classifier. Shadow attacks are
optional and require user-provided shadow artifact directories.

The core reports are `summary.json`, `scores.csv` and `attack_details.json`.
Inspect each attack record, including null or unavailable metrics. Retain input
roles and selection-time hold provenance when comparing privacy results. No
previous results, copied synthetic data or cluster batch schedulers are included.
