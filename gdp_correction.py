"""GDP drifter correction utility.

Applies two corrections to observed drifter velocity:
1) Wind-slippage correction using ERA5 10 m wind (u10, v10).
2) Stokes-drift correction:
   - Preferred: use wave Stokes velocity (ust, vst) dataset.
   - Fallback (when no Stokes dataset is available): estimate from wind with
     U_stokes ≈ gamma * U10, V_stokes ≈ gamma * V10.

Corrected Eulerian current is estimated as:
    u_corr = u_obs - alpha * u10 - u_stokes
    v_corr = v_obs - alpha * v10 - v_stokes

Typical alpha values are around 0.005~0.015 depending on drogue state.
Common wind-based Stokes approximation uses gamma=0.015.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr

DEFAULT_GDP_PATH = r"D:\download\uswc_drifter_6hour_2023.nc"
DEFAULT_ERA5_PATH = r"D:\data\wind_2023_uswc.nc"
DEFAULT_OUT_PATH = r"D:\download\uswc_drifter_6hour_2023_corrected.nc"

LON_CANDIDATES = ["lon", "longitude", "LON", "LONGITUDE", "x"]
LAT_CANDIDATES = ["lat", "latitude", "LAT", "LATITUDE", "y"]
TIME_CANDIDATES = ["time", "Time", "TIME", "date", "datetime"]
U_CANDIDATES = ["u", "ve", "u_eastward", "eastward_velocity", "u_current"]
V_CANDIDATES = ["v", "vn", "v_northward", "northward_velocity", "v_current"]


def _normalize_lon(lon_series: pd.Series, target_lon: xr.DataArray) -> pd.Series:
    """Match GDP longitude convention to forcing dataset convention."""
    target_min = float(target_lon.min())
    target_max = float(target_lon.max())
    lon = lon_series.copy()

    # Dataset uses 0..360
    if target_min >= 0.0 and target_max > 180.0:
        lon = lon % 360.0
    else:  # Dataset uses -180..180
        lon = ((lon + 180.0) % 360.0) - 180.0
    return lon


def _open_field(path: str, u_name: str, v_name: str) -> xr.Dataset:
    ds = xr.open_dataset(path)
    if u_name not in ds.variables or v_name not in ds.variables:
        raise KeyError(f"Missing variables {u_name}/{v_name} in {path}")
    return ds[[u_name, v_name]]


def _infer_coord_name(ds: xr.Dataset, candidates: list[str]) -> str:
    for c in candidates:
        if c in ds.coords:
            return c
    raise KeyError(f"None of coordinate names found: {candidates}")


def _interp_uv(
    ds: xr.Dataset,
    df: pd.DataFrame,
    u_name: str,
    v_name: str,
) -> pd.DataFrame:
    time_name = _infer_coord_name(ds, ["time", "valid_time"])
    lat_name = _infer_coord_name(ds, ["lat", "latitude"])
    lon_name = _infer_coord_name(ds, ["lon", "longitude"])

    tmp = df.copy()
    tmp["lon_norm"] = _normalize_lon(tmp["lon"], ds[lon_name])

    # xarray interp over datetime coords expects datetime-like arrays, not
    # timezone-aware Python objects. Convert to tz-naive UTC datetime64.
    interp_time = pd.to_datetime(tmp["time"], utc=True).dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")

    out = ds.interp(
        {
            time_name: xr.DataArray(interp_time, dims="points"),
            lat_name: xr.DataArray(tmp["lat"].to_numpy(), dims="points"),
            lon_name: xr.DataArray(tmp["lon_norm"].to_numpy(), dims="points"),
        },
        method="linear",
    )

    return pd.DataFrame(
        {
            "u": out[u_name].to_numpy(),
            "v": out[v_name].to_numpy(),
        }
    )


def correct_gdp(
    gdp_df: pd.DataFrame,
    era5_ds: xr.Dataset,
    stokes_ds: Optional[xr.Dataset],
    alpha_wind: float,
    beta_stokes: float,
    gdp_u_col: str,
    gdp_v_col: str,
    era5_u_col: str,
    era5_v_col: str,
    stokes_u_col: str,
    stokes_v_col: str,
    stokes_from_wind_coeff: float,
) -> pd.DataFrame:
    df = gdp_df.copy()
    # Keep timestamps in UTC but strip timezone for xarray datetime interpolation.
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)

    era5_uv = _interp_uv(era5_ds, df, era5_u_col, era5_v_col)
    df["u10"] = era5_uv["u"]
    df["v10"] = era5_uv["v"]

    if stokes_ds is not None:
        stokes_uv = _interp_uv(stokes_ds, df, stokes_u_col, stokes_v_col)
        df["ust"] = stokes_uv["u"]
        df["vst"] = stokes_uv["v"]
    else:
        # Empirical fallback when no Stokes dataset is available:
        # U_stokes ≈ gamma * U10, V_stokes ≈ gamma * V10.
        df["ust"] = stokes_from_wind_coeff * df["u10"]
        df["vst"] = stokes_from_wind_coeff * df["v10"]

    df["u_wind_slip"] = alpha_wind * df["u10"]
    df["v_wind_slip"] = alpha_wind * df["v10"]
    df["u_stokes"] = beta_stokes * df["ust"]
    df["v_stokes"] = beta_stokes * df["vst"]

    df["u_corr"] = df[gdp_u_col] - df["u_wind_slip"] - df["u_stokes"]
    df["v_corr"] = df[gdp_v_col] - df["v_wind_slip"] - df["v_stokes"]

    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correct GDP drifter data for wind slip and Stokes drift")
    p.add_argument("--gdp", default=DEFAULT_GDP_PATH, help="Input GDP CSV/Parquet/NetCDF")
    p.add_argument("--era5", default=DEFAULT_ERA5_PATH, help="ERA5 netCDF path")
    p.add_argument(
        "--stokes",
        default=None,
        help="Stokes drift netCDF path (optional). If omitted, use gamma*U10 approximation.",
    )
    p.add_argument("--out", default=DEFAULT_OUT_PATH, help="Output CSV/Parquet/NetCDF path")

    p.add_argument("--gdp-u-col", default="u", help="GDP u velocity column")
    p.add_argument("--gdp-v-col", default="v", help="GDP v velocity column")
    p.add_argument("--gdp-lon-col", default="lon", help="GDP longitude column/variable name")
    p.add_argument("--gdp-lat-col", default="lat", help="GDP latitude column/variable name")
    p.add_argument("--gdp-time-col", default="time", help="GDP time column/variable name")
    p.add_argument("--era5-u-col", default="u10", help="ERA5 u10 variable name")
    p.add_argument("--era5-v-col", default="v10", help="ERA5 v10 variable name")
    p.add_argument("--stokes-u-col", default="ust", help="Stokes u variable name")
    p.add_argument("--stokes-v-col", default="vst", help="Stokes v variable name")

    p.add_argument("--alpha-wind", type=float, default=0.007, help="Wind-slip coefficient")
    p.add_argument("--beta-stokes", type=float, default=1.0, help="Stokes scaling coefficient")
    p.add_argument(
        "--stokes-from-wind-coeff",
        type=float,
        default=0.015,
        help="Gamma in U_stokes≈gamma*U10 when --stokes is not provided",
    )

    return p.parse_args()


def _read_gdp(path: str, lon_col: str, lat_col: str, time_col: str, u_col: str, v_col: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif ext == ".nc":
        ds = xr.open_dataset(path)
        df_all = ds.to_dataframe().reset_index()
        cols = set(df_all.columns)

        def pick_name(preferred: str, candidates: list[str], label: str) -> str:
            if preferred in cols:
                return preferred
            for c in candidates:
                if c in cols:
                    return c
            raise KeyError(
                f"GDP netCDF cannot find {label}. "
                f"Tried preferred='{preferred}' and candidates={candidates}. "
                f"Available columns={sorted(cols)}"
            )

        lon_name = pick_name(lon_col, LON_CANDIDATES, "longitude")
        lat_name = pick_name(lat_col, LAT_CANDIDATES, "latitude")
        time_name = pick_name(time_col, TIME_CANDIDATES, "time")
        u_name = pick_name(u_col, U_CANDIDATES, "u velocity")
        v_name = pick_name(v_col, V_CANDIDATES, "v velocity")

        df = df_all[[lon_name, lat_name, time_name, u_name, v_name]].copy()
        lon_col, lat_col, time_col, u_col, v_col = lon_name, lat_name, time_name, u_name, v_name
    else:
        raise ValueError(f"Unsupported GDP extension: {ext}")

    rename_map = {
        lon_col: "lon",
        lat_col: "lat",
        time_col: "time",
        u_col: "u",
        v_col: "v",
    }
    missing_in_df = [k for k in rename_map if k not in df.columns]
    if missing_in_df:
        raise KeyError(f"GDP input missing columns after loading: {missing_in_df}")
    df = df.rename(columns=rename_map)
    return df[["lon", "lat", "time", "u", "v"]]


def _write_table(df: pd.DataFrame, path: str) -> None:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
        return
    if ext in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    if ext == ".nc":
        xr.Dataset.from_dataframe(df).to_netcdf(path)
        return
    raise ValueError(f"Unsupported output extension: {ext}")


def main() -> None:
    args = parse_args()

    gdp = _read_gdp(
        path=args.gdp,
        lon_col=args.gdp_lon_col,
        lat_col=args.gdp_lat_col,
        time_col=args.gdp_time_col,
        u_col=args.gdp_u_col,
        v_col=args.gdp_v_col,
    )
    print(f"Loaded GDP rows: {len(gdp)} from {args.gdp}")

    era5 = _open_field(args.era5, args.era5_u_col, args.era5_v_col)
    stokes = None
    if args.stokes:
        stokes = _open_field(args.stokes, args.stokes_u_col, args.stokes_v_col)

    corrected = correct_gdp(
        gdp_df=gdp,
        era5_ds=era5,
        stokes_ds=stokes,
        alpha_wind=args.alpha_wind,
        beta_stokes=args.beta_stokes,
        gdp_u_col="u",
        gdp_v_col="v",
        era5_u_col=args.era5_u_col,
        era5_v_col=args.era5_v_col,
        stokes_u_col=args.stokes_u_col,
        stokes_v_col=args.stokes_v_col,
        stokes_from_wind_coeff=args.stokes_from_wind_coeff,
    )

    _write_table(corrected, args.out)
    print(f"Saved corrected GDP data to: {args.out}")


if __name__ == "__main__":
    main()
