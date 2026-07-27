# Ingesting raw WAM data into truth-data artifacts

Converts raw WAM `.nc` output into `physics_avg_csv` (`avg_density_vs_time`) and
`physics_model_hourly_npz` (`lonlat_snapshot_series`). Raw WAM output is never queried live.

## Standalone primitive usage

`validation/wam_convert.py` is AWS-agnostic — local `.nc` file(s) in, truth artifact out:

```python
from rope_dev_tools.validation.wam_convert import convert_avg_density_csv, convert_hourly_npz

convert_avg_density_csv(
    ["wam.20240511_000000.nc", "wam.20240511_010000.nc"],
    "wam_avg_density_2024.csv", altitudes_km=[250.0, 300.0, 400.0],
)

convert_hourly_npz(sorted_hourly_files, "wam_density_2023_01_01.npz", altitudes_km=[400.0])
```

- One WAM timestep per raw file expected; a multi-timestep file works too.
- `avg_density_csv` (grid-mean) is order-invariant. `hourly_npz` converts WAM's longitude grid into
  ROPE's LST grid via a per-timestep circular shift (`LST = (utc_hour + lon_deg/15) mod 24`) —
  requires a uniformly spaced longitude axis, else raises `ValueError`.

## Variable-name overrides

Defaults (`_DEFAULT_VARIABLE_NAMES` in `wam_convert.py`): `time`, `lon`, `lat`, `hlevs` (altitude,
km), `den` (density, kg/m³). Inspect a file that doesn't match:

```python
import xarray as xr
print(xr.open_dataset("some_file.nc"))
```

Override:

```python
convert_avg_density_csv(paths, out_path, altitudes_km=[...], variable_names={"density": "rho"})
```

A non-matching name raises `WamVariableNotFoundError` listing every available variable/dim.

## Fetching from S3 or a local mirror

`validation/wam_source.py` resolves one UTC hourly timestamp to a local `.nc` path: `S3WamSource`
or `LocalMirrorWamSource` (offline archive, same raw files on disk instead of S3).

Config: `rope-data/validation/wam_sources.json`, one entry per year (folder naming isn't
consistent across years):

```json
{
  "default_filename_pattern": "wam_fixed_height.wam.%Y%m%d_%H0000.nc",
  "s3": {
    "bucket": "my-bucket",
    "years": { "2013": { "prefix": "some/prefix/for/2013/" } }
  },
  "offline": {
    "years": { "2013": { "dir": "/mnt/wam_archive/2013" } }
  }
}
```

- `filename_pattern` is a `strftime` template; override per year if needed.
- Only hourly (`_HH0000.nc`) files are ever requested.

## Running the pipeline

```
python scripts/build_validation_data.py --suite rope-data/validation/validation-wam-v1.json \
    --out-dir rope-data/validation --source s3 --source-config rope-data/validation/wam_sources.json

python scripts/build_validation_data.py --suite ... --out-dir ... \
    --source offline --source-config rope-data/validation/wam_sources.json
```

`--only-check <id>` (repeatable) restricts to specific checks.

## How dedup/cleanup works

`wam_ingest.py` merges by output filename: checks/periods referencing the same
`physics_avg_csv`/`physics_model_hourly_npz` union their altitude and timestamp requirements into
one target. Every distinct raw timestamp across the whole suite is fetched exactly once, used by
every target that needs it, then deleted. Keep filenames consistent across checks covering the
same year — inconsistent names fetch that year twice.

## Not yet covered

`satellite_orbit_density`/`doy_lat_orbit_density`'s `physics_model_track_csv` — no satellite
truth-data source decided yet.
