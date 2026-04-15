"""
FlowCast v2 — CMIP6 Basin-Clipped Downloader
=============================================
Downloads CMIP6 variables for the Cauvery basin ONLY, using your shapefile
to clip the spatial domain before any data is pulled to disk.

Variables downloaded (mapped to pipeline names):
  CMIP6 name  →  pipeline name   unit conversion
  ─────────────────────────────────────────────────
  pr          →  precip          kg/m²/s × 86400  = mm/day
  tasmax      →  tmax            K - 273.15       = °C
  tasmin      →  tmin            K - 273.15       = °C
  rsds        →  radiation       W/m² (kept as-is)
  hurs        →  vapor_pressure  RH% → hPa via Tetens

Models (from config.py):
  MRI-ESM2-0, CMCC-ESM2, MPI-ESM1-2-LR, INM-CM5-0, ACCESS-CM2

Scenarios: historical | ssp245 | ssp585

Source: Pangeo / Google Cloud CMIP6 (public, no auth required)
Catalog: https://storage.googleapis.com/cmip6/pangeo-cmip6.json

Usage:
  python cmip6_downloader.py --shapefile path/to/cauvery.shp

Output structure:
  cmip6_downloads/
  ├── historical/
  │   ├── MRI-ESM2-0_historical_daily.csv
  │   └── ...
  ├── ssp245/
  │   ├── MRI-ESM2-0_ssp245_daily.csv
  │   └── ...
  └── ssp585/
      ├── MRI-ESM2-0_ssp585_daily.csv
      └── ...

Each CSV has columns: date, precip, tmax, tmin, radiation, vapor_pressure
"""

import argparse
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

warnings.filterwarnings("ignore")

# ── Try importing cloud-access libraries ──────────────────────────────────
try:
    import intake
    import gcsfs
    HAS_INTAKE = True
except ImportError:
    HAS_INTAKE = False

try:
    import fsspec
    HAS_FSSPEC = True
except ImportError:
    HAS_FSSPEC = False


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Pangeo public CMIP6 catalog (no credentials required)
PANGEO_CATALOG = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"

# Variables to download: {CMIP6 variable_id: pipeline column name}
VARIABLES = {
    "pr":     "precip",
    "tasmax": "tmax",
    "tasmin": "tmin",
    "rsds":   "radiation",
    "hurs":   "vapor_pressure",  # relative humidity → converted to hPa
}

# Models from config.py
MODELS = [
    "MRI-ESM2-0",
    "CMCC-ESM2",
    "MPI-ESM1-2-LR",
    "INM-CM5-0",
    "ACCESS-CM2",
]

# Scenarios
SCENARIOS = {
    "historical": ("1990-01-01", "2020-12-31"),
    "ssp245":     ("2025-01-01", "2100-12-31"),
    "ssp585":     ("2025-01-01", "2100-12-31"),
}

# Output directory
OUT_DIR = Path("cmip6_downloads")


# ═══════════════════════════════════════════════════════════════════════════
#  UNIT CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════════

def convert_units(da: xr.DataArray, varname: str) -> xr.DataArray:
    """Convert CMIP6 units to pipeline-ready units."""
    if varname == "pr":
        # kg/m²/s  →  mm/day
        da = da * 86400.0
        da.attrs["units"] = "mm/day"

    elif varname in ("tasmax", "tasmin"):
        # Kelvin  →  Celsius
        da = da - 273.15
        da.attrs["units"] = "degC"

    elif varname == "hurs":
        # Relative humidity (%)  →  actual vapour pressure (hPa)
        # Uses Tetens formula with mean temp approximation
        # e_s(T) = 6.1078 × exp(17.27 × T / (T + 237.3))
        # e_a    = RH/100 × e_s
        # We store the RH-fraction here; proper conversion done
        # after tmin is available (see post_process_df)
        da.attrs["_needs_temp_for_vp"] = True
        da.attrs["units"] = "%"

    # rsds (radiation) is already in W/m², kept as-is

    return da


def rh_to_vapor_pressure(rh_pct: np.ndarray, tmin_c: np.ndarray) -> np.ndarray:
    """
    Convert relative humidity (%) to actual vapour pressure (hPa).
    Tetens formula using daily minimum temperature as dew-point proxy.

    RH ≈ 100 × e_a / e_s(Tmin)  (standard meteorological approximation)
    → e_a = RH/100 × e_s(Tmin)
    """
    es_tmin = 6.1078 * np.exp(17.27 * tmin_c / (tmin_c + 237.3))
    return (rh_pct / 100.0) * es_tmin


# ═══════════════════════════════════════════════════════════════════════════
#  SHAPEFILE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def load_basin_geometry(shapefile_path: str):
    """
    Load basin shapefile and return:
      - gdf: GeoDataFrame (WGS84)
      - bbox: (lon_min, lat_min, lon_max, lat_max)
      - union: single union geometry for masking
    """
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    union = gdf.geometry.union_all()
    lon_min, lat_min, lon_max, lat_max = union.bounds

    # Add a small buffer to ensure boundary cells are included
    buffer = 0.25  # degrees
    bbox = (
        lon_min - buffer,
        lat_min - buffer,
        lon_max + buffer,
        lat_max + buffer,
    )

    print(f"  Basin bounding box: lon [{bbox[0]:.2f}, {bbox[2]:.2f}]  "
          f"lat [{bbox[1]:.2f}, {bbox[3]:.2f}]")
    print(f"  Basin area (approx): {union.area * 111**2:.0f} km²")

    return gdf, bbox, union


def clip_to_basin(ds: xr.Dataset, bbox: tuple, union_geom) -> xr.Dataset:
    """
    Clip xarray Dataset to basin bounding box, then mask to basin polygon.

    Args:
        ds: xarray Dataset with lat/lon coords (could be named lat/lon or
            latitude/longitude depending on the model)
        bbox: (lon_min, lat_min, lon_max, lat_max)
        union_geom: shapely geometry of the basin

    Returns:
        Spatially-clipped Dataset
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    # Normalise coordinate names (some models use 'latitude'/'longitude')
    rename_map = {}
    if "latitude" in ds.coords and "lat" not in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)

    # Handle 0-360 longitude grids (some models use 0-360 instead of -180-180)
    if float(ds.lon.min()) >= 0 and float(ds.lon.max()) > 180:
        # Convert to -180 to 180
        ds = ds.assign_coords(lon=(ds.lon + 180) % 360 - 180)
        ds = ds.sortby("lon")

    # Bounding box clip
    ds = ds.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
    )

    # If lat is stored in descending order
    if float(ds.lat[0]) > float(ds.lat[-1]):
        ds = ds.sel(lat=slice(lat_max, lat_min))

    return ds


def spatial_mean_over_basin(ds: xr.Dataset, union_geom, varname: str) -> xr.DataArray:
    """
    Compute area-weighted spatial mean over the basin polygon.
    Uses cosine(lat) weighting.

    For simplicity (and speed), we use a bounding-box mean weighted
    by cos(lat) here. For exact basin-area masking, use rioxarray
    clip (enabled when rioxarray is available).
    """
    da = ds[varname]

    try:
        import rioxarray  # optional but gives exact masking
        da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.clip([union_geom], crs="EPSG:4326", drop=True)
    except ImportError:
        pass  # fall back to bbox mean

    # Cosine-latitude weighting
    weights = np.cos(np.deg2rad(da.lat))
    da_weighted = da.weighted(weights)
    return da_weighted.mean(dim=["lat", "lon"])


# ═══════════════════════════════════════════════════════════════════════════
#  PANGEO CATALOG SEARCH
# ═══════════════════════════════════════════════════════════════════════════

def load_pangeo_catalog():
    """Load the Pangeo CMIP6 intake-esm catalog."""
    if not HAS_INTAKE:
        raise ImportError(
            "intake and intake-esm are required.\n"
            "Install: pip install intake intake-esm gcsfs"
        )
    print("  Loading Pangeo CMIP6 catalog...")
    col = intake.open_esm_datastore(PANGEO_CATALOG)
    print(f"  Catalog loaded: {len(col.df)} entries")
    return col


def search_catalog(col, model: str, scenario: str, variable: str) -> object:
    """
    Search the Pangeo catalog for a specific model/scenario/variable.

    Returns an intake-esm catalog subset, or None if not found.
    """
    experiment_map = {
        "historical": "historical",
        "ssp245":     "ssp245",
        "ssp585":     "ssp585",
    }

    subset = col.search(
        source_id=model,
        experiment_id=experiment_map[scenario],
        variable_id=variable,
        table_id="day",          # daily data
        member_id="r1i1p1f1",   # first ensemble member
    )

    if len(subset.df) == 0:
        # Try alternative member IDs
        for member in ["r1i1p1f2", "r2i1p1f1", "r1i1p1f3"]:
            subset = col.search(
                source_id=model,
                experiment_id=experiment_map[scenario],
                variable_id=variable,
                table_id="day",
                member_id=member,
            )
            if len(subset.df) > 0:
                break

    if len(subset.df) == 0:
        return None

    return subset


# ═══════════════════════════════════════════════════════════════════════════
#  DOWNLOAD PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def download_variable(
    col,
    model: str,
    scenario: str,
    variable: str,
    bbox: tuple,
    union_geom,
    date_start: str,
    date_end: str,
) -> pd.Series:
    """
    Download one variable for one model × scenario, clipped to the basin.

    Returns: pd.Series indexed by date, values in pipeline units.
    """
    subset = search_catalog(col, model, scenario, variable)
    if subset is None:
        print(f"    ✗ Not found in catalog: {model}/{scenario}/{variable}")
        return None

    print(f"    ↓ Downloading {variable} | {model} | {scenario} ...")

    try:
        dsets = subset.to_dataset_dict(
            xarray_open_kwargs={
                "consolidated": True,
                "decode_times": True,
            },
            storage_options={"token": "anon"},  # public bucket
        )
    except Exception as e:
        print(f"    ✗ Download error: {e}")
        return None

    if not dsets:
        return None

    # Take first available dataset
    ds = list(dsets.values())[0]

    # Clip spatially to basin
    ds = clip_to_basin(ds, bbox, union_geom)

    # Time slice
    try:
        ds = ds.sel(time=slice(date_start, date_end))
    except Exception:
        pass

    if len(ds.time) == 0:
        print(f"    ✗ No data in time range {date_start}–{date_end}")
        return None

    # Spatial mean
    da_basin = spatial_mean_over_basin(ds, union_geom, variable)

    # Unit conversion
    da_basin = convert_units(da_basin, variable)

    # Load to memory and convert to pandas Series
    values = da_basin.load().values
    times = pd.to_datetime(da_basin.time.values)

    series = pd.Series(values, index=times, name=variable)
    series.index.name = "date"

    print(f"    ✓ {variable}: {len(series)} days  "
          f"(mean={series.mean():.2f}, max={series.max():.2f})")

    return series


def download_model_scenario(
    col,
    model: str,
    scenario: str,
    bbox: tuple,
    union_geom,
    out_dir: Path,
    force_redownload: bool = False,
) -> pd.DataFrame | None:
    """
    Download all variables for one model × scenario combination.
    Saves to out_dir/{scenario}/{model}_{scenario}_daily.csv
    """
    out_path = out_dir / scenario / f"{model}_{scenario}_daily.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force_redownload:
        print(f"  [SKIP] Already exists: {out_path.name}")
        return pd.read_csv(out_path, index_col="date", parse_dates=True)

    date_start, date_end = SCENARIOS[scenario]
    print(f"\n  ── {model} / {scenario}  ({date_start} → {date_end}) ──")

    series_dict = {}

    for cmip_var, pipeline_col in VARIABLES.items():
        s = download_variable(
            col, model, scenario, cmip_var,
            bbox, union_geom, date_start, date_end,
        )
        if s is not None:
            series_dict[cmip_var] = s

    if not series_dict:
        print(f"  ✗ No data downloaded for {model}/{scenario}")
        return None

    # Combine all variables into one DataFrame
    df = pd.DataFrame(series_dict)
    df.index.name = "date"

    # Align to a common daily index (fill gaps with NaN then interpolate)
    full_idx = pd.date_range(date_start, date_end, freq="D")
    df = df.reindex(full_idx)
    df = df.interpolate(method="time", limit=7)  # fill ≤7-day gaps
    df = df.dropna()

    # Convert column names to pipeline names
    rename = {k: v for k, v in VARIABLES.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Post-process: convert relative humidity to actual vapour pressure
    if "vapor_pressure" in df.columns and "tmin" in df.columns:
        df["vapor_pressure"] = rh_to_vapor_pressure(
            df["vapor_pressure"].values, df["tmin"].values
        )
        print(f"    ✓ vapor_pressure converted from RH% to hPa")

    # Ensure non-negative precip
    if "precip" in df.columns:
        df["precip"] = df["precip"].clip(lower=0.0)

    # Save
    df.to_csv(out_path)
    print(f"  ✓ Saved: {out_path}  ({len(df)} rows, {len(df.columns)} cols)")

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  FORECAST CSV GENERATOR  (matches your existing forecast format)
# ═══════════════════════════════════════════════════════════════════════════

def generate_forecast_csvs(
    out_dir: Path,
    models: list[str] = None,
    scenarios: list[str] = ("ssp245", "ssp585"),
):
    """
    From the downloaded per-model CSVs, create ensemble-mean forecast CSVs
    that match your existing forecast_input_ssp245.csv / ssp585.csv format.

    Output columns: date, precip, tmax, tmin, radiation, vapor_pressure,
                    precip_std (std across GCMs = uncertainty envelope)

    These CSVs are what your pipeline's CMIP6Projector reads for delta-change.
    """
    models = models or MODELS

    for scenario in scenarios:
        dfs = []
        for model in models:
            fpath = out_dir / scenario / f"{model}_{scenario}_daily.csv"
            if fpath.exists():
                df = pd.read_csv(fpath, index_col="date", parse_dates=True)
                dfs.append(df)

        if not dfs:
            print(f"  No data found for {scenario} — skipping forecast CSV")
            continue

        # Align all models to common date range
        common_idx = dfs[0].index
        for d in dfs[1:]:
            common_idx = common_idx.intersection(d.index)

        aligned = [d.reindex(common_idx) for d in dfs]

        # Ensemble mean and std across GCMs
        stack = np.stack([d.values for d in aligned], axis=0)
        mean_df = pd.DataFrame(
            np.nanmean(stack, axis=0),
            index=common_idx,
            columns=aligned[0].columns,
        )
        std_precip = pd.Series(
            np.nanstd(stack[:, :, aligned[0].columns.tolist().index("precip")], axis=0),
            index=common_idx,
            name="precip_std",
        )

        forecast_df = mean_df.copy()
        forecast_df["precip_std"] = std_precip
        forecast_df.index.name = "date"

        # Save in format matching your existing forecast_input_ssp*.csv
        # Rename precip → rainfall_max_mm, precip_std → rainfall_std_mm
        # (to stay compatible with current pipeline)
        compat_df = forecast_df.rename(columns={
            "precip":     "rainfall_max_mm",
            "precip_std": "rainfall_std_mm",
        })

        out_path = out_dir / f"forecast_input_{scenario}.csv"
        compat_df.to_csv(out_path)
        print(f"\n  ✓ Forecast CSV saved: {out_path}")
        print(f"    Columns: {list(compat_df.columns)}")
        print(f"    Period:  {common_idx[0].date()} → {common_idx[-1].date()}")
        print(f"    Models merged: {len(dfs)}")

        # Also save the FULL multi-variable version for the pipeline's
        # CMIP6Projector (which needs tmax, tmin, radiation, vapor_pressure)
        full_path = out_dir / f"forecast_full_{scenario}.csv"
        forecast_df.to_csv(full_path)
        print(f"    Full-variable CSV: {full_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  DELTA-CHANGE SUMMARY  (feeds directly into cmip6.py)
# ═══════════════════════════════════════════════════════════════════════════

def compute_delta_factors(out_dir: Path, models: list[str] = None) -> dict:
    """
    Compute delta-change factors between historical and future periods.
    This replaces the synthetic delta factors in CMIP6Projector.generate_synthetic_gcm_data()

    Returns a dict in the exact format that cmip6.py expects:
        {
          "MRI-ESM2-0": {
            "ssp245_2031_2050": {
              "tmax_delta": float,   # °C change
              "tmin_delta": float,   # °C change
              "precip_factors": {1: float, 2: float, ..., 12: float}
            },
            ...
          },
          ...
        }
    """
    models = models or MODELS
    projections = {}

    future_periods = [
        ("2031-01-01", "2050-12-31", 2031, 2050),
        ("2051-01-01", "2070-12-31", 2051, 2070),
        ("2071-01-01", "2090-12-31", 2071, 2090),
    ]

    for model in models:
        hist_path = out_dir / "historical" / f"{model}_historical_daily.csv"
        if not hist_path.exists():
            print(f"  ✗ No historical data for {model}, skipping delta calc")
            continue

        hist = pd.read_csv(hist_path, index_col="date", parse_dates=True)
        hist_monthly = hist.groupby(hist.index.month).mean()

        projections[model] = {}

        for ssp in ("ssp245", "ssp585"):
            ssp_path = out_dir / ssp / f"{model}_{ssp}_daily.csv"
            if not ssp_path.exists():
                continue

            future_all = pd.read_csv(ssp_path, index_col="date", parse_dates=True)

            for date_start, date_end, yr_start, yr_end in future_periods:
                future = future_all.loc[date_start:date_end]
                if future.empty:
                    continue

                future_monthly = future.groupby(future.index.month).mean()

                key = f"{ssp}_{yr_start}_{yr_end}"

                # Temperature deltas (°C)
                tmax_delta = (
                    future_monthly["tmax"] - hist_monthly["tmax"]
                ).mean() if "tmax" in future_monthly else 0.0

                tmin_delta = (
                    future_monthly["tmin"] - hist_monthly["tmin"]
                ).mean() if "tmin" in future_monthly else 0.0

                # Monthly precipitation change factors
                precip_factors = {}
                for month in range(1, 13):
                    hist_p = hist_monthly.loc[month, "precip"] if "precip" in hist_monthly else 1.0
                    fut_p = future_monthly.loc[month, "precip"] if month in future_monthly.index else hist_p
                    factor = fut_p / (hist_p + 1e-6)
                    precip_factors[month] = round(float(np.clip(factor, 0.3, 3.0)), 3)

                projections[model][key] = {
                    "tmax_delta":     round(float(tmax_delta), 2),
                    "tmin_delta":     round(float(tmin_delta), 2),
                    "precip_factors": precip_factors,
                }

                print(f"  {model}/{key}: "
                      f"Δtmax={tmax_delta:+.2f}°C  "
                      f"Δtmin={tmin_delta:+.2f}°C  "
                      f"precip_avg_factor={np.mean(list(precip_factors.values())):.3f}")

    # Save for use by cmip6.py
    import json
    delta_path = out_dir / "delta_factors.json"
    with open(delta_path, "w") as f:
        json.dump(projections, f, indent=2)
    print(f"\n  ✓ Delta factors saved: {delta_path}")
    print(f"    → Pass this file to CMIP6Projector instead of generate_synthetic_gcm_data()")

    return projections


# ═══════════════════════════════════════════════════════════════════════════
#  INSTALLATION HELPER
# ═══════════════════════════════════════════════════════════════════════════

def check_and_install_dependencies():
    """Check required packages and print install command if missing."""
    required = {
        "intake":       "intake",
        "intake_esm":   "intake-esm",
        "gcsfs":        "gcsfs",
        "geopandas":    "geopandas",
        "xarray":       "xarray",
        "fsspec":       "fsspec",
    }
    optional = {
        "rioxarray":    "rioxarray",     # exact basin masking
        "zarr":         "zarr",          # fast cloud reads
        "dask":         "dask[array]",   # lazy loading
    }

    missing_required = []
    missing_optional = []

    for module, pkg in required.items():
        try:
            __import__(module)
        except ImportError:
            missing_required.append(pkg)

    for module, pkg in optional.items():
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(pkg)

    if missing_required:
        print("\n  ✗ Missing REQUIRED packages:")
        print(f"    pip install {' '.join(missing_required)}")
        return False

    if missing_optional:
        print(f"\n  ⚠ Optional packages not installed (recommended):")
        print(f"    pip install {' '.join(missing_optional)}")

    return True


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download CMIP6 data clipped to Cauvery basin shapefile"
    )
    parser.add_argument(
        "--shapefile", "-s",
        required=True,
        help="Path to Cauvery basin shapefile (.shp or .geojson or .gpkg)",
    )
    parser.add_argument(
        "--out-dir", "-o",
        default="cmip6_downloads",
        help="Output directory (default: cmip6_downloads/)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="CMIP6 model IDs to download (default: all 5 from config.py)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["historical", "ssp245", "ssp585"],
        help="Scenarios to download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if CSV already exists",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check dependencies, don't download",
    )
    args = parser.parse_args()

    print("\n" + "═" * 65)
    print("  FlowCast v2 — CMIP6 Basin-Clipped Downloader")
    print("═" * 65)

    # 1. Dependency check
    print("\n  Checking dependencies...")
    ok = check_and_install_dependencies()
    if not ok:
        sys.exit(1)
    if args.check_only:
        print("\n  All required dependencies present.")
        sys.exit(0)

    # 2. Load basin shapefile
    print(f"\n  Loading shapefile: {args.shapefile}")
    gdf, bbox, union_geom = load_basin_geometry(args.shapefile)

    # 3. Load Pangeo catalog
    print("\n  Connecting to Pangeo CMIP6 catalog...")
    col = load_pangeo_catalog()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # 4. Download each model × scenario
    print(f"\n  Downloading {len(args.models)} models × {len(args.scenarios)} scenarios")
    print(f"  Variables: {', '.join(VARIABLES.keys())}")

    downloaded = {}
    failed = []

    for model in args.models:
        for scenario in args.scenarios:
            print(f"\n{'─'*65}")
            df = download_model_scenario(
                col=col,
                model=model,
                scenario=scenario,
                bbox=bbox,
                union_geom=union_geom,
                out_dir=out_dir,
                force_redownload=args.force,
            )
            if df is not None:
                downloaded[f"{model}/{scenario}"] = df
            else:
                failed.append(f"{model}/{scenario}")

    # 5. Generate ensemble forecast CSVs
    print("\n" + "═" * 65)
    print("  Generating ensemble forecast CSVs...")
    generate_forecast_csvs(
        out_dir=out_dir,
        models=args.models,
        scenarios=[s for s in args.scenarios if s != "historical"],
    )

    # 6. Compute delta factors
    if "historical" in args.scenarios:
        print("\n" + "═" * 65)
        print("  Computing delta-change factors (historical vs future)...")
        delta_factors = compute_delta_factors(out_dir=out_dir, models=args.models)

    # 7. Summary
    print("\n" + "═" * 65)
    print("  DOWNLOAD SUMMARY")
    print("═" * 65)
    print(f"  ✓ Downloaded: {len(downloaded)} combinations")
    if failed:
        print(f"  ✗ Failed:     {len(failed)} combinations")
        for f in failed:
            print(f"    - {f}")
    print(f"\n  Output directory: {out_dir.resolve()}")
    print(f"""
  Next steps:
  ─────────────────────────────────────────────────────────────
  1. Replace forecast CSVs in your project:
       cp {out_dir}/forecast_input_ssp245.csv your_project/
       cp {out_dir}/forecast_input_ssp585.csv your_project/

  2. Replace synthetic delta factors in cmip6.py:
     In CMIP6Projector.generate_synthetic_gcm_data(), load from:
       {out_dir}/delta_factors.json

  3. For pipeline training, the historical CSVs give you real
     climate variables (tmax, tmin, radiation, vapor_pressure)
     to augment your master_dataset.csv.
  ─────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()