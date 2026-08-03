# Ingesting satellite density data (GRACE/GRACE-FO) into truth-data artifacts

Converts raw GRACE/GRACE-FO density CDF files into `satellite_track_csv`
(`satellite_orbit_density`). WAM's along-track sampling (`physics_model_track_csv`) then reads that
CSV directly — see the "Along-track satellite sampling" section in `ingesting-wam-data.md`.

## Standalone primitive usage

`validation/satellite_convert.py` is network-agnostic — local `.cdf` day file(s) in, truth artifact
out:

```python
from rope_dev_tools.validation.satellite_convert import convert_satellite_track_csv

convert_satellite_track_csv(sorted_day_files, "gracefo_2023.csv", cadence_seconds=600)
```

- One calendar day per raw file, 10-second cadence (8640 records/day).
- Rows with `validity_flag != 0` are dropped by default (`include_flagged=True` keeps them).
- Downsampled by averaging into `cadence_seconds`-wide bins (default 600 = 10 min; `0` disables).
  `lon`/`lst` are periodic — averaged with a circular mean, not a plain mean (359° and 1° average
  to 0°, not 180°). `datetime` is the mean of each bin's actual raw timestamps.
- Output columns: `datetime, lst, lat, lon, alt_km, density` — `lon` is an extra column beyond what
  `truth_data.load_truth_csv` requires, needed by WAM's along-track sampler.

## Variable-name overrides

Defaults (`_DEFAULT_VARIABLE_NAMES` in `satellite_convert.py`): `time`, `longitude`, `latitude`,
`altitude` (meters), `local_solar_time`, `density`, `validity_flag`. Inspect a file that doesn't
match:

```python
import cdflib
print(cdflib.cdfread.CDF("some_file.cdf").cdf_info().zVariables)
```

Override:

```python
convert_satellite_track_csv(paths, out_path, variable_names={"density": "rho"})
```

A non-matching name raises `SatelliteVariableNotFoundError` listing every available variable.

## Fetching from swarm-diss or a local mirror

`validation/satellite_source.py` resolves one calendar day to a local `.cdf` path:
`SwarmDissSatelliteSource` (public server, no auth) or `LocalMirrorSatelliteSource` (offline
archive).

Config: `rope-data/validation/satellite_sources.json`, one entry per year only where the default
doesn't apply:

```json
{
  "default_satellite": "Sat_1",
  "remote": {
    "years": { "2013": { "mission": "GRACE" } }
  },
  "offline": {
    "years": { "2013": { "dir": "/mnt/satellite_archive/2013", "mission": "GRACE" } }
  }
}
```

- Mission defaults by year (`< 2018` → GRACE, `>= 2018` → GRACE-FO) — override per year for edge
  cases; GRACE-FO has no `Sat_2`.
- GRACE and GRACE-FO share an identical file layout and CDF schema on the server, just a different
  mission folder and filename prefix (`GR_`/`GF_`).

## Running the pipeline

```
python scripts/build_satellite_data.py --suite rope-data/validation/validation-wam-v1.json \
    --out-dir rope-data/validation --source remote --source-config rope-data/validation/satellite_sources.json

python scripts/build_satellite_data.py --suite ... --out-dir ... \
    --source offline --source-config rope-data/validation/satellite_sources.json
```

`--only-check <id>` (repeatable) restricts to specific checks. `--cadence-seconds` overrides the
default 600s downsampling (`0` disables it). Progress prints as `[i/n] fetching <date>` by default;
pass `--quiet` to suppress it.

## How dedup/gap-handling works

`satellite_ingest.py` merges by output filename, same as the WAM pipeline. Every distinct calendar
day across the whole suite is fetched exactly once. A day missing upstream (e.g. a real GRACE
outage — confirmed for all of March 2013) is a soft, collectible failure: every *other* target still
gets fully built and written; the run ends with one `ValueError` naming exactly which dates are
missing for which output files. A target with any missing day writes no output file at all — never
a partial CSV.
