"""Build training data.nc (u_bg/v_bg/u_obs/v_obs) from CMEMS + GDP.

Outputs NetCDF variables:
- u_bg, v_bg: background flow from CMEMS on [time, lat, lon]
- u_obs, v_obs: sparse observations mapped from GDP to nearest grid/time (NaN elsewhere)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

U_CANDS = ["u_clean", "u", "uo", "eastward_sea_water_velocity", "water_u", "ugos"]
V_CANDS = ["v_clean", "v", "vo", "northward_sea_water_velocity", "water_v", "vgos"]
LON_CANDS = ["lon", "longitude"]
LAT_CANDS = ["lat", "latitude"]
TIME_CANDS = ["time", "valid_time"]


def _find(ds: xr.Dataset, cands: list[str], label: str) -> str:
    for c in cands:
        if c in ds.variables or c in ds.coords:
            return c
    raise KeyError(f"Cannot find {label}. candidates={cands}, available={list(ds.variables)}")


def _normalize_lon_to_grid(lon_vals: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
    mn, mx = float(np.nanmin(lon_grid)), float(np.nanmax(lon_grid))
    if mn >= 0 and mx > 180:
        return np.mod(lon_vals, 360.0)
    return ((lon_vals + 180.0) % 360.0) - 180.0


def _load_gdp_df(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif ext == ".nc":
        df = xr.open_dataset(path).to_dataframe().reset_index()
    else:
        raise ValueError(f"Unsupported GDP file: {path}")

    col_map = {
        "lon": ["lon", "longitude"],
        "lat": ["lat", "latitude"],
        "time": ["time", "date", "datetime"],
        "u": ["u_corr", "u_obs", "u"],
        "v": ["v_corr", "v_obs", "v"],
    }
    chosen: dict[str, str] = {}
    for key, cands in col_map.items():
        for c in cands:
            if c in df.columns:
                chosen[key] = c
                break
        if key not in chosen:
            raise KeyError(f"GDP missing {key}. columns={list(df.columns)}")

    out = pd.DataFrame(
        {
            "lon": df[chosen["lon"]].astype(float),
            "lat": df[chosen["lat"]].astype(float),
            "time": pd.to_datetime(df[chosen["time"]], utc=True, errors="coerce").dt.tz_convert(None),
            "u": df[chosen["u"]].astype(float),
            "v": df[chosen["v"]].astype(float),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def _auto_scale_obs_by_bg(u_bg: np.ndarray, v_bg: np.ndarray, u_obs: np.ndarray, v_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bg = np.abs(np.concatenate([u_bg[np.isfinite(u_bg)], v_bg[np.isfinite(v_bg)]]))
    ob = np.abs(np.concatenate([u_obs[np.isfinite(u_obs)], v_obs[np.isfinite(v_obs)]]))
    if bg.size == 0 or ob.size == 0:
        return u_obs, v_obs
    p95_bg = float(np.nanpercentile(bg, 95))
    p95_obs = float(np.nanpercentile(ob, 95))
    if p95_bg <= 1e-9:
        return u_obs, v_obs
    ratio = p95_obs / p95_bg
    if ratio > 50:
        print(f"[WARN] obs/bg p95 ratio={ratio:.2f}, auto scale obs by /100")
        return u_obs / 100.0, v_obs / 100.0
    return u_obs, v_obs


def main(args: argparse.Namespace) -> None:
    cmems = xr.open_dataset(args.cmems)
    u_name = _find(cmems, U_CANDS, "u")
    v_name = _find(cmems, V_CANDS, "v")
    lon_name = _find(cmems, LON_CANDS, "lon")
    lat_name = _find(cmems, LAT_CANDS, "lat")
    time_name = _find(cmems, TIME_CANDS, "time")

    u_bg = cmems[u_name].squeeze().transpose(time_name, lat_name, lon_name).values.astype(np.float32)
    v_bg = cmems[v_name].squeeze().transpose(time_name, lat_name, lon_name).values.astype(np.float32)
    times = pd.to_datetime(cmems[time_name].values)
    lats = cmems[lat_name].values.astype(np.float64)
    lons = cmems[lon_name].values.astype(np.float64)

    u_obs = np.full_like(u_bg, np.nan, dtype=np.float32)
    v_obs = np.full_like(v_bg, np.nan, dtype=np.float32)

    gdp = _load_gdp_df(args.gdp)
    gdp_lon = _normalize_lon_to_grid(gdp["lon"].to_numpy(np.float64), lons)
    gdp_lat = gdp["lat"].to_numpy(np.float64)
    gdp_t = pd.to_datetime(gdp["time"].to_numpy())
    gdp_u = gdp["u"].to_numpy(np.float32)
    gdp_v = gdp["v"].to_numpy(np.float32)

    times_ns = times.to_numpy(dtype="datetime64[ns]")
    max_dt = np.timedelta64(args.max_time_diff_hours, "h")

    used = 0
    for lo, la, tt, uu, vv in zip(gdp_lon, gdp_lat, gdp_t.to_numpy(dtype="datetime64[ns]"), gdp_u, gdp_v):
        if not np.isfinite([lo, la, uu, vv]).all():
            continue
        ti = int(np.argmin(np.abs(times_ns - tt)))
        if np.abs(times_ns[ti] - tt) > max_dt:
            continue
        yi = int(np.argmin(np.abs(lats - la)))
        xi = int(np.argmin(np.abs(lons - lo)))
        u_obs[ti, yi, xi] = uu
        v_obs[ti, yi, xi] = vv
        used += 1

    u_obs, v_obs = _auto_scale_obs_by_bg(u_bg, v_bg, u_obs, v_obs)
    sp = np.sqrt(np.nan_to_num(u_obs, nan=0.0) ** 2 + np.nan_to_num(v_obs, nan=0.0) ** 2)
    mask = np.isfinite(u_obs) & np.isfinite(v_obs) & (sp <= args.max_obs_speed)
    u_obs = np.where(mask, u_obs, np.nan).astype(np.float32)
    v_obs = np.where(mask, v_obs, np.nan).astype(np.float32)

    out = xr.Dataset(
        data_vars={
            "u_bg": ((time_name, lat_name, lon_name), np.nan_to_num(u_bg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)),
            "v_bg": ((time_name, lat_name, lon_name), np.nan_to_num(v_bg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)),
            "u_obs": ((time_name, lat_name, lon_name), u_obs),
            "v_obs": ((time_name, lat_name, lon_name), v_obs),
        },
        coords={time_name: cmems[time_name].values, lat_name: lats, lon_name: lons},
        attrs={
            "source_cmems": str(args.cmems),
            "source_gdp": str(args.gdp),
            "max_time_diff_hours": args.max_time_diff_hours,
            "max_obs_speed": args.max_obs_speed,
            "gdp_points_total": int(len(gdp)),
            "gdp_points_mapped": int(used),
            "gdp_points_valid_after_filter": int(mask.sum()),
        },
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(args.out)
    print(f"[INFO] Saved: {args.out}")
    print(f"[INFO] mapped={used}, valid_after_filter={int(mask.sum())}, total_gdp={len(gdp)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build data.nc with u_bg/v_bg/u_obs/v_obs from CMEMS + GDP")
    p.add_argument("--cmems", default=r"D:/data/uswc_2023_cmems_cleaned.nc")
    p.add_argument("--gdp", default=r"D:/download/uswc_drifter_6hour_2023_corrected.nc")
    p.add_argument("--out", default=r"D:/data/output/train_data_2023.nc")
    p.add_argument("--max-time-diff-hours", type=int, default=6)
    p.add_argument("--max-obs-speed", type=float, default=5.0)
    main(p.parse_args())
