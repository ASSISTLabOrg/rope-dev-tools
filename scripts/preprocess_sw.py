#!/usr/bin/env python3
"""
preprocess_sw.py — converts a CelesTrak CSV or CSSI TXT space-weather file to the ROPE .swbin binary driver format.
Format auto-detected from file extension or content.

Usage
~~~~~
  python scripts/preprocess_sw.py --input SW-All.csv --output data/sw.swbin
  python scripts/preprocess_sw.py --url https://celestrak.org/SpaceData/SW-All.csv --output data/sw.swbin
  python scripts/preprocess_sw.py --input SpaceWeather-All.txt --output data/sw.swbin --no-predicted
  python scripts/preprocess_sw.py --input SW-All.csv --output drivers.csv --output-format csv
"""

from __future__ import annotations

import argparse
import io
import ssl
import struct
import sys
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


# ---------------------------------------------------------------------------
# .swbin constants (must match include/rope/io/driver_bin.h)
# ---------------------------------------------------------------------------
_SWBIN_MAGIC   = 0x52505357  # "RPSW"
_SWBIN_VERSION = 1

# section names after BEGIN/END markers in the .txt format
_TXT_SECTION_NAMES = {"OBSERVED", "DAILY_PREDICTED", "MONTHLY_PREDICTED"}


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _download(url: str) -> str:
    """Download *url* and return the raw text.  Falls back to no-verify SSL."""
    req = urllib.request.Request(url, headers={"User-Agent": "rope-preprocess/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode("utf-8")
    except urllib.error.URLError as e:
        if "SSL" not in str(e) and "certificate" not in str(e).lower():
            raise
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
            return r.read().decode("utf-8")


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_input_format(content: str) -> str:
    """Return 'csv' or 'txt' by inspecting the first meaningful line."""
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(("DATATYPE", "VERSION", "UPDATED", "NUM_", "BEGIN")):
            return "txt"
        if s.startswith("DATE"):
            return "csv"
        break
    return "csv"


# ---------------------------------------------------------------------------
# CelesTrak CSV parser
# ---------------------------------------------------------------------------

def _parse_celestrak_content(raw: str) -> pd.DataFrame:
    """Parses CelesTrak CSV content into a daily DataFrame indexed by datetime, columns F10.7_OBS and KP1..KP8."""
    lines = [
        ln for ln in raw.splitlines()
        if (s := ln.strip()) and (s.startswith("DATE") or s[0].isdigit())
    ]
    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        parse_dates=["DATE"],
    )
    df = df.rename(columns={"DATE": "datetime"})
    df = df.set_index("datetime").sort_index()
    return df


def load_celestrak(source: str, *, is_url: bool = False) -> pd.DataFrame:
    """Load a CelesTrak SW CSV from a local path or URL."""
    if is_url:
        raw = _download(source)
    else:
        with open(source, encoding="utf-8") as f:
            raw = f.read()
    return _parse_celestrak_content(raw)


# ---------------------------------------------------------------------------
# CSSI fixed-width TXT parser
# ---------------------------------------------------------------------------

def _parse_txt_rows(lines: list[str]) -> list[dict]:
    """Parses data rows from one section of the CSSI .txt file (whitespace-split; year/month/day at 0-2, KP1..KP8 at 5-12, F10.7_Obs at 30)."""
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) < 31:
            continue
        try:
            year  = int(parts[0])
            month = int(parts[1])
            day   = int(parts[2])
            kp    = [float(parts[i]) for i in range(5, 13)]
            f107  = float(parts[30])
        except (ValueError, IndexError):
            continue
        rows.append({
            "datetime": pd.Timestamp(year=year, month=month, day=day),
            "F10.7_OBS": f107,
            **{f"KP{i + 1}": kp[i] for i in range(8)},
        })
    return rows


def _parse_cssi_txt_content(raw: str, *,
                             include_predicted: bool = True) -> pd.DataFrame:
    """Parses CSSI .txt content into a daily DataFrame with the same schema as _parse_celestrak_content()."""
    current: str | None = None
    sections: dict[str, list[str]] = {}

    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("BEGIN "):
            sec = s[6:].strip()
            if sec in _TXT_SECTION_NAMES:
                current = sec
                sections.setdefault(current, [])
        elif s.startswith("END "):
            current = None
        elif current is not None and s[0].isdigit():
            sections[current].append(s)

    want = ["OBSERVED"]
    if include_predicted:
        want += ["DAILY_PREDICTED", "MONTHLY_PREDICTED"]

    all_rows: list[dict] = []
    for sec in want:
        if sec in sections:
            all_rows.extend(_parse_txt_rows(sections[sec]))

    if not all_rows:
        raise ValueError(
            "No data rows found in the .txt file. "
            "Check that the file contains a BEGIN OBSERVED section."
        )

    df = pd.DataFrame(all_rows).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]  # drop section-boundary dupes
    return df


def load_cssi_txt(source: str, *, is_url: bool = False,
                  include_predicted: bool = True) -> pd.DataFrame:
    """Load a CSSI SpaceWeather .txt file from a local path or URL."""
    if is_url:
        raw = _download(source)
    else:
        with open(source, encoding="utf-8") as f:
            raw = f.read()
    return _parse_cssi_txt_content(raw, include_predicted=include_predicted)


# ---------------------------------------------------------------------------
# Format-agnostic loader
# ---------------------------------------------------------------------------

def load_sw(source: str, *, is_url: bool = False,
            fmt: str = "auto",
            include_predicted: bool = True) -> pd.DataFrame:
    """Loads space-weather data, auto-detecting CSV vs TXT format if needed."""
    # format from extension, if possible
    if fmt == "auto" and not is_url:
        if source.lower().endswith(".txt"):
            fmt = "txt"
        elif source.lower().endswith(".csv"):
            fmt = "csv"

    # detect format from content if still unknown
    if is_url:
        raw = _download(source)
        if fmt == "auto":
            fmt = _detect_input_format(raw[:2048])
    else:
        with open(source, encoding="utf-8") as fh:
            raw = fh.read()
        if fmt == "auto":
            fmt = _detect_input_format(raw[:2048])

    if fmt == "txt":
        return _parse_cssi_txt_content(raw, include_predicted=include_predicted)
    else:
        return _parse_celestrak_content(raw)


# ---------------------------------------------------------------------------
# Core conversion (format-independent)
# ---------------------------------------------------------------------------

def convert(df: pd.DataFrame) -> pd.DataFrame:
    """Converts a daily space-weather DataFrame to an hourly ROPE driver DataFrame (datetime, f10, kp, doy, hour)."""
    # 1. interpolate F10.7 daily -> hourly via PCHIP
    f107_series = (
        pd.to_numeric(df["F10.7_OBS"], errors="coerce")
        .dropna()
        .sort_index()
    )
    t0 = f107_series.index[0]
    x_f107 = (f107_series.index - t0).total_seconds().to_numpy() / 3600.0
    y_f107 = f107_series.to_numpy(dtype=float)

    t_hourly = pd.date_range(
        f107_series.index.min(),
        f107_series.index.max() + pd.Timedelta(hours=23),
        freq="h",
    )
    xq_f107 = (t_hourly - t0).total_seconds().to_numpy() / 3600.0
    yq_f107 = PchipInterpolator(x_f107, y_f107, extrapolate=False)(xq_f107)

    f107_hourly = (
        pd.DataFrame({"datetime": t_hourly, "f10": yq_f107})
        .dropna()
        .reset_index(drop=True)
    )

    # 2. interpolate Kp 3-hourly -> hourly via PCHIP, then /10
    kp_cols = [c for c in df.columns if c.startswith("KP") and c != "KP_SUM"]
    kp_cols = sorted(kp_cols, key=lambda c: int(c[2:]))  # KP1 < KP2 < … < KP8

    kp_long = (
        df[kp_cols]
        .apply(pd.to_numeric, errors="coerce")
        .stack()
        .reset_index()
        .rename(columns={"level_1": "kp_col", 0: "kp_raw"})
    )
    kp_long["kp_idx"] = kp_long["kp_col"].str.extract(r"KP(\d+)").astype(int) - 1
    kp_long["t3h"] = kp_long["datetime"] + pd.to_timedelta(
        kp_long["kp_idx"] * 3, unit="h"
    )
    kp_long["day"] = kp_long["datetime"].dt.floor("D")

    first_kp_per_day = (
        kp_long.sort_values(["day", "t3h"])
        .groupby("day")["kp_raw"]
        .first()
    )
    next_day_kp0 = first_kp_per_day.shift(-1)

    hourly_parts = []
    for day in next_day_kp0.index[:-1]:
        g = kp_long[kp_long["day"] == day].sort_values("t3h")
        x = (g["t3h"] - day).dt.total_seconds().to_numpy() / 3600.0
        y = g["kp_raw"].to_numpy(dtype=float)
        if len(x) < 2 or not np.all(np.isfinite(y)):
            continue
        y24 = float(next_day_kp0.loc[day])
        if not np.isfinite(y24):
            continue
        x = np.r_[x, 24.0]
        y = np.r_[y, y24]
        order = np.argsort(x)
        x, y = x[order], y[order]
        xq = np.arange(24.0)
        yq = PchipInterpolator(x, y, extrapolate=False)(xq)
        hourly_parts.append(
            pd.DataFrame(
                {
                    "datetime": day + pd.to_timedelta(xq, unit="h"),
                    "kp": yq / 10.0,
                }
            )
        )

    kp_hourly = pd.concat(hourly_parts, ignore_index=True)

    # 3. inner merge
    merged = pd.merge(f107_hourly, kp_hourly, on="datetime", how="inner")
    merged = merged.dropna(subset=["f10", "kp"]).reset_index(drop=True)

    # 4. add time features
    merged["doy"]  = merged["datetime"].dt.day_of_year
    merged["hour"] = merged["datetime"].dt.hour

    return merged[["datetime", "f10", "kp", "doy", "hour"]]


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _to_unix(dt_series: pd.Series) -> np.ndarray:
    """Convert a UTC-naive DatetimeSeries to integer Unix timestamps."""
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    tz_aware = dt_series.dt.tz_localize("UTC")
    return ((tz_aware - epoch).dt.total_seconds()).to_numpy(dtype=np.int64)


def write_swbin(path: str, df: pd.DataFrame) -> None:
    """Writes the converted DataFrame to a ROPE .swbin file (16-byte header + 24-byte records; see include/rope/io/driver_bin.h)."""
    nrows      = len(df)
    timestamps = _to_unix(df["datetime"])
    f10_vals   = df["f10"].to_numpy(dtype=np.float32)
    kp_vals    = df["kp"].to_numpy(dtype=np.float32)
    doy_vals   = (df["doy"] + df["hour"] / 24.0).to_numpy(dtype=np.float32)
    hour_vals  = df["hour"].to_numpy(dtype=np.int32)

    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", _SWBIN_MAGIC, _SWBIN_VERSION, nrows, 0))
        for tp, f10, kp, doy, hr in zip(timestamps, f10_vals, kp_vals,
                                         doy_vals, hour_vals):
            f.write(struct.pack("<qfffi", int(tp), float(f10), float(kp),
                                float(doy), int(hr)))

    print(f"Wrote {nrows:,} rows → {path}")


def write_csv(path: str, df: pd.DataFrame) -> None:
    """Write the converted DataFrame to a ROPE-format CSV (datetime, f10, kp)."""
    out = df[["datetime", "f10", "kp"]].copy()
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)
    print(f"Wrote {len(out):,} rows → {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a CelesTrak CSV or CSSI TXT space-weather file "
            "to ROPE .swbin driver format."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", metavar="FILE",
                     help="Local space-weather file (.csv or .txt)")
    src.add_argument("--url", metavar="URL",
                     help="URL to download the space-weather file from")

    parser.add_argument("--output", required=True, metavar="FILE",
                        help="Output file (.swbin or .csv)")
    parser.add_argument("--input-format", choices=["auto", "csv", "txt"],
                        default="auto", dest="input_format",
                        help="Input format ('auto' detects from extension then content)")
    parser.add_argument("--output-format", choices=["swbin", "csv"],
                        dest="output_format",
                        help="Output format (default: inferred from --output extension)")
    parser.add_argument("--no-predicted", action="store_true",
                        help="TXT only: exclude DAILY_PREDICTED/MONTHLY_PREDICTED, keep OBSERVED only")

    args = parser.parse_args(argv)

    include_predicted = not args.no_predicted

    # load
    if args.url:
        print(f"Downloading {args.url} …")
        df_raw = load_sw(args.url, is_url=True,
                         fmt=args.input_format,
                         include_predicted=include_predicted)
    else:
        print(f"Loading {args.input} …")
        df_raw = load_sw(args.input, is_url=False,
                         fmt=args.input_format,
                         include_predicted=include_predicted)

    print(f"  {len(df_raw):,} daily rows  "
          f"[{df_raw.index.min().date()} → {df_raw.index.max().date()}]")

    # convert
    print("Interpolating …")
    df_out = convert(df_raw)
    print(f"  {len(df_out):,} hourly rows after merge")

    # write
    out_fmt = args.output_format
    if out_fmt is None:
        out_fmt = "swbin" if args.output.endswith(".swbin") else "csv"

    if out_fmt == "swbin":
        write_swbin(args.output, df_out)
    else:
        write_csv(args.output, df_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
