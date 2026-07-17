from __future__ import annotations

import numpy as np

from postprocess.validator import TabularValidator


def test_categorical_integer_float_matches_string_legal_value() -> None:
    schema_card = {
        "column_order": ["SEX", "target"],
        "columns": {
            "SEX": {"type": "categorical", "legal_values": ["1", "2"]},
            "target": {"type": "categorical", "legal_values": ["0", "1"]},
        },
    }
    stats_card = {
        "categorical_top_values": {
            "SEX": [{"value": "1"}],
            "target": [{"value": "0"}],
        }
    }
    validator = TabularValidator(schema_card, stats_card)

    bundle = validator.validate(
        [
            {
                "candidate_id": 7,
                "row": {"SEX": np.float64(2.0), "target": 1.0},
            }
        ]
    )

    assert bundle.report["num_valid"] == 1
    assert bundle.report["num_rejected"] == 0
    assert bundle.valid_records[0]["row"] == {"SEX": "2", "target": "1"}
    assert bundle.report["repair_action_histogram"] == {
        "category_numeric_normalize:SEX": 1,
        "category_numeric_normalize:target": 1,
    }
