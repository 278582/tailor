from __future__ import annotations

from types import SimpleNamespace

from post_selection_tool.pareto_repair import dcr_signal_schema_card


def test_dcr_signal_full_reference_overrides_theta_col_ps() -> None:
    config = SimpleNamespace(
        dcr_signal_full_reference=True,
        theta_col_ps_all_columns=False,
        theta_default_fidelity_columns=False,
        theta_guidance_report={"enabled": True},
    )
    selector = SimpleNamespace(
        schema_card={"columns": {"a": {"type": "numerical"}, "target": {"type": "categorical"}}},
        privacy_columns=["a"],
        feature_columns=["a"],
        target_column="target",
    )

    schema_card = dcr_signal_schema_card(config, selector)

    assert schema_card["dcr_signal_column_source"] == "full_reference_override"
    assert "dcr_signal_column_order" not in schema_card
    assert "dcr_signal_full_reference_ignored_reason" not in schema_card
