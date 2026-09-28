from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarWeightVector:
    name: str
    fidelity: float
    privacy: float
    utility: float

    @property
    def fidelity_1d(self) -> float:
        return 0.5 * float(self.fidelity)

    @property
    def fidelity_2d(self) -> float:
        return 0.5 * float(self.fidelity)

    def to_report(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "scalar_fidelity_weight": float(self.fidelity),
            "scalar_privacy_weight": float(self.privacy),
            "scalar_utility_weight": float(self.utility),
            "scalar_fidelity_1d_weight": float(self.fidelity_1d),
            "scalar_fidelity_2d_weight": float(self.fidelity_2d),
        }


RECOMMENDED_SCALAR_WEIGHT_VECTORS: tuple[ScalarWeightVector, ...] = (
    ScalarWeightVector("balanced", 0.50, 0.30, 0.20),
    ScalarWeightVector("fidelity", 0.70, 0.20, 0.10),
    ScalarWeightVector("privacy", 0.40, 0.50, 0.10),
    ScalarWeightVector("utility", 0.40, 0.20, 0.40),
)

PRIMARY_SCALAR_KEEP_FILENAME = "selection_scalar.csv"


def scalar_variant_keep_filename(name: str) -> str:
    return f"selection_scalar_{name}.csv"


def resolve_scalar_weight_vectors(
    *,
    enabled: bool,
    fidelity_weight: float,
    privacy_weight: float,
    utility_weight: float,
) -> tuple[ScalarWeightVector, ...]:
    if not enabled:
        return (
            ScalarWeightVector(
                "balanced",
                float(fidelity_weight),
                float(privacy_weight),
                float(utility_weight),
            ),
        )
    return RECOMMENDED_SCALAR_WEIGHT_VECTORS
