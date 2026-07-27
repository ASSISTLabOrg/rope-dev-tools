"""truth_data loaders: load_truth_csv, load_avg_density_csv (single path and multi-file list), load_ascending_track_csv."""

from __future__ import annotations

import pytest

from rope_dev_tools.validation.truth_data import (
    load_ascending_track_csv,
    load_avg_density_csv,
    load_truth_csv,
)


def test_load_truth_csv_round_trip(tmp_path):
    path = tmp_path / "truth.csv"
    path.write_text("datetime,lst,lat,alt_km,density\n2024-01-01 00:00:00,12.0,0.0,400.0,1.0e-12\n")
    df = load_truth_csv(path)
    assert list(df["density"]) == [1.0e-12]


def test_load_truth_csv_missing_column_raises(tmp_path):
    path = tmp_path / "truth.csv"
    path.write_text("datetime,lst,lat,density\n2024-01-01 00:00:00,12.0,0.0,1.0e-12\n")
    with pytest.raises(ValueError, match="alt_km"):
        load_truth_csv(path)


def test_load_ascending_track_csv_requires_ascending_column(tmp_path):
    path = tmp_path / "truth.csv"
    path.write_text("datetime,lst,lat,alt_km,density\n2024-01-01 00:00:00,12.0,0.0,400.0,1.0e-12\n")
    with pytest.raises(ValueError, match="ascending"):
        load_ascending_track_csv(path)

    path.write_text(
        "datetime,lst,lat,alt_km,density,ascending\n2024-01-01 00:00:00,12.0,0.0,400.0,1.0e-12,1\n"
    )
    df = load_ascending_track_csv(path)
    assert list(df["ascending"]) == [1]


def test_load_avg_density_csv_single_path(tmp_path):
    path = tmp_path / "truth.csv"
    path.write_text("datetime,alt_km,density\n2024-01-01 01:00:00,400.0,1.1e-12\n2024-01-01 00:00:00,400.0,1.0e-12\n")
    df = load_avg_density_csv(path)
    # sorted by datetime even though the file itself wasn't
    assert list(df["density"]) == [1.0e-12, 1.1e-12]


def test_load_avg_density_csv_missing_column_raises(tmp_path):
    path = tmp_path / "truth.csv"
    path.write_text("datetime,density\n2024-01-01 00:00:00,1.0e-12\n")
    with pytest.raises(ValueError, match="alt_km"):
        load_avg_density_csv(path)


def test_load_avg_density_csv_list_of_paths_concatenates_sorted(tmp_path):
    path_a = tmp_path / "a.csv"
    path_a.write_text("datetime,alt_km,density\n2024-01-02 00:00:00,400.0,2.0e-12\n")
    path_b = tmp_path / "b.csv"
    path_b.write_text("datetime,alt_km,density\n2024-01-01 00:00:00,400.0,1.0e-12\n")

    df = load_avg_density_csv([path_a, path_b])
    assert list(df["density"]) == [1.0e-12, 2.0e-12]


def test_load_avg_density_csv_list_missing_column_names_offending_file(tmp_path):
    path_a = tmp_path / "a.csv"
    path_a.write_text("datetime,alt_km,density\n2024-01-01 00:00:00,400.0,1.0e-12\n")
    path_b = tmp_path / "b.csv"
    path_b.write_text("datetime,density\n2024-01-02 00:00:00,2.0e-12\n")

    with pytest.raises(ValueError, match=r"b\.csv"):
        load_avg_density_csv([path_a, path_b])
