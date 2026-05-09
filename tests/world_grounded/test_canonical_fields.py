import numpy as np

from scripts.canonicalize_wham_fixed_beta import trim_sequence_fields


def test_trim_sequence_fields_preserves_contact_and_feet():
    record = {
        "contact": np.ones((5, 4), dtype=np.float64),
        "feet_world": np.ones((5, 4, 3), dtype=np.float64) * 2.0,
        "unrelated": "keep",
    }

    trim_sequence_fields(record, 3)

    assert record["contact"].shape == (3, 4)
    assert record["feet_world"].shape == (3, 4, 3)
    assert record["contact"].dtype == np.float32
    assert record["feet_world"].dtype == np.float32
    assert record["unrelated"] == "keep"
