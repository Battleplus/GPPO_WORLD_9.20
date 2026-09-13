import math

from tools.evaluate_m10_consequence_quality import public_physics
from tools.generate_m10_consequence_dataset import make_split
from gppo_world.consequence_data import example_from_dict


def test_public_physics_baseline_uses_only_serialized_graph_context():
    rows, _ = make_split(
        "train",
        1,
        92001,
        2,
        6,
        prefix_policy="public-hash-legal",
    )
    example = example_from_dict(next(row for row in rows if row["target"]["action"] != 24), expected_horizon_steps=6, strict_identity=True)
    prediction = public_physics(example)
    assert set(prediction) == {"travel_time", "service_progress", "energy_delta", "deadline_risk"}
    assert all(math.isfinite(value) for value in prediction.values())
    assert prediction["travel_time"] >= 0
    assert 0 <= prediction["deadline_risk"] <= 1


def test_counterfactual_energy_scope_is_system_wide():
    rows, _ = make_split(
        "train",
        1,
        92001,
        2,
        6,
        prefix_policy="public-hash-legal",
    )
    candidate = next(row for row in rows if row["target"]["action"] != 24)
    noop = next(row for row in rows if row["target"]["action"] == 24)
    assert candidate["label_provenance"]["outcome_scope"] == "selected-uav-task + system-energy"
    assert noop["label_provenance"]["outcome_scope"] == "system-energy-only"
