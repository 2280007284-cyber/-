from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

U_CANDS = ["u", "uo", "u_clean", "u_hr", "u_recon", "eastward_sea_water_velocity"]
V_CANDS = ["v", "vo", "v_clean", "v_hr", "v_recon", "northward_sea_water_velocity"]
LON_CANDS = ["lon", "longitude"]
LAT_CANDS = ["lat", "latitude"]
TIME_CANDS = ["time", "valid_time"]


def set_style() -> None:
    mpl.rcParams.update(
        {
            "savefig.dpi": 400,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def _find(ds: xr.Dataset, cands: list[str], label: str) -> str:
    for c in cands:
        if c in ds.variables or c in ds.coords:
            return c
    raise KeyError(f"Cannot find {label}. candidates={cands}, available={list(ds.variables)}")


def _open(path: str | Path) -> xr.Dataset:
    try:
        return xr.open_dataset(path)
    except ValueError:
        return xr.open_dataset(path, engine="scipy")


def _scale_to_mps(x: np.ndarray, units: str = "") -> np.ndarray:
    u = units.lower()
    if "cm/s" in u or "cm s-1" in u or "cm s^-1" in u:
        return x / 100.0
    if "m/day" in u or "m d-1" in u:
        return x / 86400.0
    return x


def _normalize_lon(lon: np.ndarray, target_lon: xr.DataArray) -> np.ndarray:
    mn, mx = float(target_lon.min()), float(target_lon.max())
    if mn >= 0 and mx > 180:
        return np.mod(lon, 360.0)
    return ((lon + 180.0) % 360.0) - 180.0


def _to_hf_grid(src: xr.Dataset, hf: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    su = _find(src, U_CANDS, "u")
    sv = _find(src, V_CANDS, "v")
    slon = _find(src, LON_CANDS, "lon")
    slat = _find(src, LAT_CANDS, "lat")
    stime = _find(src, TIME_CANDS, "time")
    hlu = _find(hf, U_CANDS, "u")
    hlv = _find(hf, V_CANDS, "v")
    hllon = _find(hf, LON_CANDS, "lon")
    hllat = _find(hf, LAT_CANDS, "lat")
    hlt = _find(hf, TIME_CANDS, "time")

    lon_hf = _normalize_lon(hf[hllon].values, src[slon])
    out = src.interp(
        {
            slon: xr.DataArray(lon_hf, dims=hf[hllon].dims),
            slat: xr.DataArray(hf[hllat].values, dims=hf[hllat].dims),
            stime: xr.DataArray(hf[hlt].values.astype("datetime64[ns]"), dims=hf[hlt].dims),
        },
        method="linear",
    )
    u_src = out[su].to_numpy()
    v_src = out[sv].to_numpy()
    u_hf = hf[hlu].to_numpy()
    v_hf = hf[hlv].to_numpy()
    u_src = _scale_to_mps(u_src, str(src[su].attrs.get("units", "")))
    v_src = _scale_to_mps(v_src, str(src[sv].attrs.get("units", "")))
    u_hf = _scale_to_mps(u_hf, str(hf[hlu].attrs.get("units", "")))
    v_hf = _scale_to_mps(v_hf, str(hf[hlv].attrs.get("units", "")))
    return (u_src, v_src), (u_hf, v_hf)


def _draw_quiver(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, u: np.ndarray, v: np.ndarray, step: int) -> None:
    yy = np.arange(0, u.shape[0], step)
    xx = np.arange(0, u.shape[1], step)
    lon2d, lat2d = np.meshgrid(lon, lat)
    ax.quiver(
        lon2d[np.ix_(yy, xx)],
        lat2d[np.ix_(yy, xx)],
        u[np.ix_(yy, xx)],
        v[np.ix_(yy, xx)],
        color="k",
        scale=25,
        width=0.002,
    )


def main(args: argparse.Namespace) -> None:
    set_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kan = _open(args.recon_kan)
    mlp = _open(args.recon_mlp)
    cmems = _open(args.cmems)
    hf_files = sorted(Path(args.hf_dir).glob("*.nc"))
    if not hf_files:
        raise FileNotFoundError(f"No HF files under: {args.hf_dir}")

    # choose top-N by finite obs count
    rec = []
    for f in hf_files:
        hf = _open(f)
        hu = _find(hf, U_CANDS, "u")
        hv = _find(hf, V_CANDS, "v")
        ht = _find(hf, TIME_CANDS, "time")
        count = int(np.isfinite(hf[hu].to_numpy()).sum() + np.isfinite(hf[hv].to_numpy()).sum())
        t = pd.to_datetime(hf[ht].values.ravel()[0]) if hf[ht].values.size else pd.Timestamp("1970-01-01")
        rec.append((f, t, count))

    order = sorted(rec, key=lambda x: x[2], reverse=True)
    picked: list[tuple[Path, pd.Timestamp, int]] = []
    for item in order:
        if item[2] <= 0:
            continue
        if all(abs((item[1] - p[1]).total_seconds()) >= args.min_gap_hours * 3600 for p in picked):
            picked.append(item)
        if len(picked) >= args.top_n:
            break
    if not picked:
        raise ValueError("No snapshot time selected.")

    pd.DataFrame({"file": [p[0].name for p in picked], "time": [p[1] for p in picked], "obs_points": [p[2] for p in picked]}).to_csv(
        out_dir / "selected_times.csv", index=False
    )

    for f, t, obs in picked:
        hf = _open(f)
        (u_kan, v_kan), (u_hf, v_hf) = _to_hf_grid(kan, hf)
        (u_mlp, v_mlp), _ = _to_hf_grid(mlp, hf)
        (u_cm, v_cm), _ = _to_hf_grid(cmems, hf)
        lon_name = _find(hf, LON_CANDS, "lon")
        lat_name = _find(hf, LAT_CANDS, "lat")
        lon = hf[lon_name].values
        lat = hf[lat_name].values

        # squeeze time dim if present
        if u_kan.ndim == 3:
            u_kan, v_kan = u_kan[0], v_kan[0]
        if u_mlp.ndim == 3:
            u_mlp, v_mlp = u_mlp[0], v_mlp[0]
        if u_cm.ndim == 3:
            u_cm, v_cm = u_cm[0], v_cm[0]
        if u_hf.ndim == 3:
            u_hf, v_hf = u_hf[0], v_hf[0]

        sp_hf = np.sqrt(u_hf**2 + v_hf**2)
        sp_kan = np.sqrt(u_kan**2 + v_kan**2)
        sp_mlp = np.sqrt(u_mlp**2 + v_mlp**2)
        sp_cm = np.sqrt(u_cm**2 + v_cm**2)
        vmax = np.nanpercentile(np.concatenate([sp_hf.ravel(), sp_kan.ravel(), sp_mlp.ravel(), sp_cm.ravel()]), 99)

        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), constrained_layout=True)
        for ax, sp, u, v, title in [
            (axes[0, 0], sp_hf, u_hf, v_hf, "HF (truth)"),
            (axes[0, 1], sp_kan, u_kan, v_kan, "KAN recon"),
            (axes[1, 0], sp_mlp, u_mlp, v_mlp, "MLP recon"),
            (axes[1, 1], sp_cm, u_cm, v_cm, "CMEMS"),
        ]:
            im = ax.pcolormesh(lon, lat, sp, shading="auto", cmap="turbo", vmin=0, vmax=vmax)
            _draw_quiver(ax, lon, lat, u, v, args.quiver_step)
            ax.set_title(title)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Speed (m/s)")

        fig.suptitle(f"Snapshot {pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S')} | obs={obs}")
        stem = out_dir / f"snapshot_{pd.Timestamp(t).strftime('%Y%m%d_%H%M%S')}"
        fig.savefig(stem.with_suffix(".png"))
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)
        print(f"[INFO] saved: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="按 HF 自动选时刻，输出 KAN/MLP/CMEMS/HF 四宫格快照")
    p.add_argument("--hf-dir", default=r"D:\生产力\py程序\pythonProject\hf\202307_uswc_1km_rtv_sio")
    p.add_argument("--recon-kan", default=r"D:\data\recon_kan.nc")
    p.add_argument("--recon-mlp", default=r"D:\data\recon_mlp.nc")
    p.add_argument("--cmems", default=r"D:\data\uswc_2023_cmems_cleaned.nc")
    p.add_argument("--out-dir", default=r"D:\data\snapshots_kan_mlp_cmems_hf")
    p.add_argument("--top-n", type=int, default=6)
    p.add_argument("--min-gap-hours", type=int, default=24)
    p.add_argument("--quiver-step", type=int, default=8)
    main(p.parse_args())
