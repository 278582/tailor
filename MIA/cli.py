from __future__ import annotations

import argparse
from pathlib import Path

from .audit import AuditConfig, audit_run, audit_selection
from .attacks import make_attack_data
from .io import load_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Membership inference attack audit for tabular synthetic data.",
    )
    run_group = parser.add_argument_group("artifact run mode")
    run_group.add_argument("--run-dir", type=Path, default=None, help="Postprocess artifact run directory.")
    run_group.add_argument("--all-selections", action="store_true", help="Audit every selection_*.csv in versions/.")
    run_group.add_argument("--selection-name", type=str, default=None, help="Selection name under versions/.")
    run_group.add_argument("--selection-csv", type=Path, default=None, help="Explicit synthetic CSV to audit.")

    explicit_group = parser.add_argument_group("explicit CSV mode")
    explicit_group.add_argument("--train-csv", type=Path, default=None, help="Member/original training CSV.")
    explicit_group.add_argument("--control-csv", type=Path, default=None, help="Non-member/control CSV.")
    explicit_group.add_argument("--reference-csv", type=Path, default=None, help="Reference CSV for density calibration.")
    explicit_group.add_argument("--synthetic-csv", type=Path, default=None, help="Released synthetic CSV.")

    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for MIA reports.")
    parser.add_argument("--columns", type=str, default=None, help="Comma-separated attack columns.")
    parser.add_argument("--exclude-target", action="store_true", help="Exclude target_column from selection_context.json.")
    parser.add_argument("--reference-split", choices=["test", "holdout"], default="test")
    parser.add_argument("--density-k", type=int, default=5)
    parser.add_argument("--max-attribute-columns", type=int, default=20)
    parser.add_argument("--max-member-rows", type=int, default=0, help="Optional member sample cap; 0 means full data.")
    parser.add_argument("--max-nonmember-rows", type=int, default=0, help="Optional non-member sample cap; 0 means full data.")
    parser.add_argument("--max-synthetic-rows", type=int, default=0, help="Optional synthetic sample cap; 0 means full data.")
    parser.add_argument("--max-reference-rows", type=int, default=0, help="Optional reference sample cap; 0 means full data.")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--no-supervised-profile", action="store_true")
    parser.add_argument(
        "--shadow-run-dir",
        type=Path,
        action="append",
        default=[],
        help="Optional shadow artifact run directory with matching selection names.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> AuditConfig:
    columns = None
    if args.columns:
        columns = [column.strip() for column in args.columns.split(",") if column.strip()]
    return AuditConfig(
        seed=args.seed,
        density_k=args.density_k,
        max_attribute_columns=args.max_attribute_columns,
        max_member_rows=args.max_member_rows,
        max_nonmember_rows=args.max_nonmember_rows,
        max_synthetic_rows=args.max_synthetic_rows,
        max_reference_rows=args.max_reference_rows,
        reference_split=args.reference_split,
        exclude_target=args.exclude_target,
        columns=columns,
        include_supervised_profile=not args.no_supervised_profile,
        shadow_run_dirs=list(args.shadow_run_dir or []),
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    if args.run_dir is not None:
        summary = audit_run(
            run_dir=args.run_dir,
            out_dir=args.out_dir,
            config=config,
            all_selections=args.all_selections,
            selection_csv=args.selection_csv,
            selection_name=args.selection_name,
        )
        print(f"Audited {summary['selection_count']} selection(s). Summary: {args.out_dir / 'summary.json'}")
        return

    required = [args.train_csv, args.control_csv, args.reference_csv, args.synthetic_csv]
    if any(path is None for path in required):
        raise SystemExit("Either provide --run-dir or all explicit CSVs: --train-csv --control-csv --reference-csv --synthetic-csv")
    data = make_attack_data(
        member=load_csv(args.train_csv),
        nonmember=load_csv(args.control_csv),
        reference=load_csv(args.reference_csv),
        synthetic=load_csv(args.synthetic_csv),
        requested_columns=config.columns,
        exclude_columns=[],
    )
    report = audit_selection(
        data=data,
        out_dir=args.out_dir,
        selection_name=args.selection_name or args.synthetic_csv.stem,
        selection_csv=args.synthetic_csv,
        config=config,
        run_context={},
    )
    print(f"Audited {report['selection_name']}. Summary: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
