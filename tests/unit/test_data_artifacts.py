"""save_csv/load_csv and save_npz/load_npz round-trip through validation_data/."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rope_dev_tools.validation.data_artifacts import load_csv, load_npz, save_csv, save_npz


def test_csv_round_trip_creates_validation_data_dir(tmp_path):
    df = pd.DataFrame({"datetime": ["2024-01-01 00:00:00"], "value": [1.0]})
    relative = save_csv(tmp_path, "foo.csv", df)

    assert relative == "validation_data/foo.csv"
    assert (tmp_path / "validation_data" / "foo.csv").is_file()

    loaded = load_csv(tmp_path, relative)
    assert loaded["value"].iloc[0] == 1.0
    assert pd.api.types.is_datetime64_any_dtype(loaded["datetime"])


def test_csv_round_trip_without_datetime_column(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    relative = save_csv(tmp_path, "bar.csv", df)
    loaded = load_csv(tmp_path, relative)
    assert list(loaded["a"]) == [1, 2]


def test_npz_round_trip(tmp_path):
    relative = save_npz(tmp_path, "foo.npz", x=np.array([1.0, 2.0]), y=np.array([[1, 2], [3, 4]]))

    assert relative == "validation_data/foo.npz"
    loaded = load_npz(tmp_path, relative)
    assert set(loaded) == {"x", "y"}
    assert np.array_equal(loaded["x"], [1.0, 2.0])
    assert np.array_equal(loaded["y"], [[1, 2], [3, 4]])
