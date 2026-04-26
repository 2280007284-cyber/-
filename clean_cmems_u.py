"""CMEMS u/v cleaning/preprocessing script.

Default input:
    D:\\data\\uswc_2023_cmems.nc
Default output:
    D:\\data\\uswc_2023_cmems_cleaned.nc

Cleaning steps:
1) Auto-detect variable/coords.
2) Sort by time and remove non-finite values.
3) Remove outliers by |u|, |v| > max_abs_speed.
4) Fill missing values by interpolation (time -> lat -> lon).
5) Optional temporal smoothing.
6) Save cleaned NetCDF.
"""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

DEFAULT_IN = r"D:\data\uswc_2023_cmems.nc"
DEFAULT_OUT = r"D:\data\uswc_2023_cmems_cleaned.nc"

U_CANDIDATES = ["u", "uo", "eastward_sea_water_velocity", "water_u", "ugos"]
V_CANDIDATES = ["v", "vo", "northward_sea_water_velocity", "water_v", "vgos"]
LON_CANDIDATES = ["lon", "longitude"]
LAT_CANDIDATES = ["lat", "latitude"]
TIME_CANDIDATES = ["time", "valid_time"]


def _find_name(ds: xr.Dataset, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in ds.variables or c in ds.coords:
            return c
    raise KeyError(f"Cannot find {label}. candidates={candidates}, available={list(ds.variables)}")


def clean_cmems_u(
    in_path: str,
    out_path: str,
    max_abs_speed: float,
    interp_limit: int,
    smooth_window: int,
) -> None:
    ds = xr.open_dataset(in_path)

    u_name = _find_name(ds, U_CANDIDATES, "u variable")
    v_name = _find_name(ds, V_CANDIDATES, "v variable")
    lon_name = _find_name(ds, LON_CANDIDATES, "lon")
    lat_name = _find_name(ds, LAT_CANDIDATES, "lat")
    time_name = _find_name(ds, TIME_CANDIDATES, "time")

    u = ds[u_name].squeeze().transpose(time_name, lat_name, lon_name)
    v = ds[v_name].squeeze().transpose(time_name, lat_name, lon_name)

    # ensure monotonic time for interpolation
    u = u.sortby(time_name)
    v = v.sortby(time_name)

    # invalid -> NaN
    u = u.where(np.isfinite(u), np.nan)
    v = v.where(np.isfinite(v), np.nan)

    # outlier filter
    u = u.where(np.abs(u) <= max_abs_speed, np.nan)
    v = v.where(np.abs(v) <= max_abs_speed, np.nan)

    # fill by interpolation along time first (best for ocean time series)
    u = u.interpolate_na(dim=time_name, method="linear", limit=interp_limit, fill_value="extrapolate")
    v = v.interpolate_na(dim=time_name, method="linear", limit=interp_limit, fill_value="extrapolate")

    # then spatial dimensions
    u = u.interpolate_na(dim=lat_name, method="linear", fill_value="extrapolate")
    u = u.interpolate_na(dim=lon_name, method="linear", fill_value="extrapolate")
    v = v.interpolate_na(dim=lat_name, method="linear", fill_value="extrapolate")
    v = v.interpolate_na(dim=lon_name, method="linear", fill_value="extrapolate")

    # final fallback with nearest
    u = u.ffill(time_name).bfill(time_name)
    v = v.ffill(time_name).bfill(time_name)

    if smooth_window > 1:
        u = u.rolling({time_name: smooth_window}, center=True, min_periods=1).mean()
        v = v.rolling({time_name: smooth_window}, center=True, min_periods=1).mean()

    u = u.astype("float32")
    v = v.astype("float32")

    out_ds = xr.Dataset(
        data_vars={
            "u_clean": u,
            "v_clean": v,
        },
        coords={
            time_name: ds[time_name],
            lat_name: ds[lat_name],
            lon_name: ds[lon_name],
        },
        attrs={
            "source": in_path,
            "notes": "CMEMS cleaned u/v field for PINN-SR training",
            "max_abs_speed": max_abs_speed,
            "interp_limit": interp_limit,
            "smooth_window": smooth_window,
        },
    )

    out_ds.to_netcdf(out_path)
    print(f"Saved cleaned CMEMS to: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean CMEMS u/v NetCDF for PINN/KAN training")
    p.add_argument("--in-path", default=DEFAULT_IN)
    p.add_argument("--out-path", default=DEFAULT_OUT)
    p.add_argument("--max-abs-speed", type=float, default=3.0, help="Outlier threshold in m/s")
    p.add_argument("--interp-limit", type=int, default=8, help="Max consecutive NaNs to linearly fill in time")
    p.add_argument("--smooth-window", type=int, default=1, help="Temporal rolling mean window (1 = no smooth)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_cmems_u(
        in_path=args.in_path,
        out_path=args.out_path,
        max_abs_speed=args.max_abs_speed,
        interp_limit=args.interp_limit,
        smooth_window=args.smooth_window,
    )
