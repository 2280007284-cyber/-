"""Compare KAN reconstruction, MLP reconstruction, and CMEMS against GDP validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

U_CANDS = ["u", "uo", "u_clean", "u_hr", "u_recon", "eastward_sea_water_velocity"]
V_CANDS = ["v", "vo", "v_clean", "v_hr", "v_recon", "northward_sea_water_velocity"]
LON_CANDS = ["lon", "longitude"]
LAT_CANDS = ["lat", "latitude"]
TIME_CANDS = ["time", "valid_time"]


@dataclass
class Metrics:
    rmse_u: float
    rmse_v: float
    rmse_vec: float
    rmse_speed: float
    mae_u: float
    mae_v: float
    bias_u: float
    bias_v: float
    corr_u: float
    corr_v: float


def _find(ds: xr.Dataset, cands: list[str], label: str) -> str:
    for c in cands:
        if c in ds.variables or c in ds.coords:
            return c
    raise KeyError(f"Cannot find {label}. candidates={cands}, available={list(ds.variables)}")


def _normalize_lon(lon: np.ndarray, target_lon: xr.DataArray) -> np.ndarray:
    mn, mx = float(target_lon.min()), float(target_lon.max())
    if mn >= 0 and mx > 180:
        return np.mod(lon, 360.0)
    return ((lon + 180.0) % 360.0) - 180.0


def _pick_col(df: pd.DataFrame, key: str, cands: list[str], forced: str | None = None) -> tuple[np.ndarray, str]:
    if forced:
        if forced not in df.columns:
            raise KeyError(f"GDP missing forced column {forced}. columns={list(df.columns)}")
        return df[forced].to_numpy(), forced
    for c in cands:
        if c in df.columns:
            return df[c].to_numpy(), c
    raise KeyError(f"GDP missing {key}. candidates={cands}. columns={list(df.columns)}")


def _load_gdp(
    path: str,
    lon_col: str | None = None,
    lat_col: str | None = None,
    time_col: str | None = None,
    u_col: str | None = None,
    v_col: str | None = None,
) -> pd.DataFrame:
    ext = path.lower().split(".")[-1]
    if ext == "csv":
        df = pd.read_csv(path)
    elif ext in {"parquet", "pq"}:
        df = pd.read_parquet(path)
    elif ext == "nc":
        ds = _open_dataset_safe(path, label="GDP")
        df = ds.to_dataframe().reset_index()
    else:
        raise ValueError(f"Unsupported GDP file: {path}")

    col_map = {
        "lon": ["lon", "longitude"],
        "lat": ["lat", "latitude"],
        "time": ["time", "date", "datetime"],
        "u": ["u_corr", "u"],
        "v": ["v_corr", "v"],
    }
    out: dict[str, np.ndarray] = {}
    sel: dict[str, str] = {}
    forced_map = {"lon": lon_col, "lat": lat_col, "time": time_col, "u": u_col, "v": v_col}
    for k, cands in col_map.items():
        arr, name = _pick_col(df, k, cands, forced=forced_map[k])
        out[k], sel[k] = arr, name
    print(
        "GDP columns -> "
        f"lon:{sel['lon']} lat:{sel['lat']} time:{sel['time']} u:{sel['u']} v:{sel['v']}"
    )

    gdp = pd.DataFrame(
        {
            "lon": out["lon"].astype(np.float64),
            "lat": out["lat"].astype(np.float64),
            "time": pd.to_datetime(out["time"], utc=True).tz_convert(None),
            "u": out["u"].astype(np.float64),
            "v": out["v"].astype(np.float64),
        }
    )
    gdp = gdp.replace([np.inf, -np.inf], np.nan)
    # Common sentinel cleanup (some products use 1e20/99999-like fill values).
    bad = (np.abs(gdp["u"]) > 1e4) | (np.abs(gdp["v"]) > 1e4)
    if bad.any():
        print(f"GDP sentinel-like rows removed: {int(bad.sum())}")
        gdp.loc[bad, ["u", "v"]] = np.nan
    gdp = gdp.dropna()
    return gdp


def _open_dataset_safe(path: str, label: str) -> xr.Dataset:
    try:
        return xr.open_dataset(path)
    except ValueError as exc:
        msg = str(exc)
        if "their dependencies may not be installed" in msg:
            try:
                return xr.open_dataset(path, engine="scipy")
            except Exception as scipy_exc:  # noqa: BLE001
                raise RuntimeError(
                    f"无法打开 {label} 文件：{path}\n"
                    "检测到 xarray 后端依赖缺失（netcdf4/h5netcdf/scipy）。\n"
                    "请安装其中任一依赖，例如：\n"
                    "  pip install netCDF4\n"
                    "或把该 .nc 转成 csv/parquet 后再传入。\n"
                    f"原始错误: {exc}\nscipy重试错误: {scipy_exc}"
                ) from scipy_exc
        raise


def _split_val(gdp: pd.DataFrame, val_days: int) -> pd.DataFrame:
    cutoff = gdp["time"].max() - pd.Timedelta(days=val_days)
    return gdp[gdp["time"] >= cutoff].copy()


def _interp_to_points(ds: xr.Dataset, pts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str, str]:
    u_name = _find(ds, U_CANDS, "u")
    v_name = _find(ds, V_CANDS, "v")
    lon_name = _find(ds, LON_CANDS, "lon")
    lat_name = _find(ds, LAT_CANDS, "lat")
    time_name = _find(ds, TIME_CANDS, "time")

    p = pts.copy()
    p["lon_use"] = _normalize_lon(p["lon"].to_numpy(), ds[lon_name])

    out = ds.interp(
        {
            lon_name: xr.DataArray(p["lon_use"].to_numpy(), dims="points"),
            lat_name: xr.DataArray(p["lat"].to_numpy(), dims="points"),
            time_name: xr.DataArray(p["time"].to_numpy(dtype="datetime64[ns]"), dims="points"),
        },
        method="linear",
    )

    u = out[u_name].to_numpy()
    v = out[v_name].to_numpy()
    u_units = str(ds[u_name].attrs.get("units", "")).lower()
    v_units = str(ds[v_name].attrs.get("units", "")).lower()
    return u, v, u_units, v_units


def _scale_to_mps(x: np.ndarray, units_hint: str = "") -> tuple[np.ndarray, float]:
    u = units_hint.lower().strip()
    if "m/s" in u or "m s-1" in u or "m s^-1" in u:
        return x, 1.0
    if "cm/s" in u or "cm s-1" in u or "cm s^-1" in u:
        return x / 100.0, 0.01
    if "m/day" in u or "m d-1" in u:
        return x / 86400.0, 1.0 / 86400.0
    if "km/day" in u or "km d-1" in u:
        return x * (1000.0 / 86400.0), 1000.0 / 86400.0
    if "knot" in u or "knots" in u:
        return x * 0.514444, 0.514444
    # Unknown units: do not guess a scale factor from value magnitude.
    # Keep raw values to avoid silently introducing 100x scale errors.
    return x, 1.0


def _metrics(u_pred: np.ndarray, v_pred: np.ndarray, u_true: np.ndarray, v_true: np.ndarray) -> Metrics:
    mask = np.isfinite(u_pred) & np.isfinite(v_pred) & np.isfinite(u_true) & np.isfinite(v_true)
    up, vp, ut, vt = u_pred[mask], v_pred[mask], u_true[mask], v_true[mask]

    sp = np.sqrt(up**2 + vp**2)
    st = np.sqrt(ut**2 + vt**2)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 3:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    return Metrics(
        rmse_u=float(np.sqrt(np.mean((up - ut) ** 2))),
        rmse_v=float(np.sqrt(np.mean((vp - vt) ** 2))),
        rmse_vec=float(np.sqrt(np.mean((up - ut) ** 2 + (vp - vt) ** 2))),
        rmse_speed=float(np.sqrt(np.mean((sp - st) ** 2))),
        mae_u=float(np.mean(np.abs(up - ut))),
        mae_v=float(np.mean(np.abs(vp - vt))),
        bias_u=float(np.mean(up - ut)),
        bias_v=float(np.mean(vp - vt)),
        corr_u=corr(up, ut),
        corr_v=corr(vp, vt),
    )


def _print(name: str, m: Metrics) -> None:
    print(f"\n[{name}]")
    print(f"RMSE_u={m.rmse_u:.4f}, RMSE_v={m.rmse_v:.4f}, RMSE_vec={m.rmse_vec:.4f}, RMSE_speed={m.rmse_speed:.4f}")
    print(f"MAE_u={m.mae_u:.4f}, MAE_v={m.mae_v:.4f}")
    print(f"Bias_u={m.bias_u:.4f}, Bias_v={m.bias_v:.4f}")
    print(f"Corr_u={m.corr_u:.4f}, Corr_v={m.corr_v:.4f}")


def _brief_stats(name: str, u: np.ndarray, v: np.ndarray) -> None:
    um = np.abs(u[np.isfinite(u)])
    vm = np.abs(v[np.isfinite(v)])
    if um.size == 0 or vm.size == 0:
        print(f"{name} stats: empty")
        return
    print(
        f"{name} | |u| p50/p95=({np.nanpercentile(um,50):.4g}/{np.nanpercentile(um,95):.4g}) "
        f"|v| p50/p95=({np.nanpercentile(vm,50):.4g}/{np.nanpercentile(vm,95):.4g})"
    )


def _mask_unphysical(u: np.ndarray, v: np.ndarray, max_abs_speed: float, name: str) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(u) & np.isfinite(v) & (np.abs(u) <= max_abs_speed) & (np.abs(v) <= max_abs_speed)
    removed = np.size(u) - int(mask.sum())
    if removed > 0:
        print(f"{name}: removed {removed} points beyond ±{max_abs_speed} m/s or non-finite")
    u2 = np.where(mask, u, np.nan)
    v2 = np.where(mask, v, np.nan)
    return u2, v2


def _save_plot(m_kan: Metrics, m_mlp: Metrics, m_cmems: Metrics, out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"Skip plot: matplotlib unavailable ({exc})")
        return

    labels = ["KAN", "MLP", "CMEMS"]
    rmse_vec = [m_kan.rmse_vec, m_mlp.rmse_vec, m_cmems.rmse_vec]
    rmse_speed = [m_kan.rmse_speed, m_mlp.rmse_speed, m_cmems.rmse_speed]
    corr_u = [m_kan.corr_u, m_mlp.corr_u, m_cmems.corr_u]
    corr_v = [m_kan.corr_v, m_mlp.corr_v, m_cmems.corr_v]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].bar(labels, rmse_vec)
    axes[0, 0].set_title("RMSE_vec (lower better)")
    axes[0, 0].grid(axis="y", alpha=0.3)

    axes[0, 1].bar(labels, rmse_speed)
    axes[0, 1].set_title("RMSE_speed (lower better)")
    axes[0, 1].grid(axis="y", alpha=0.3)

    axes[1, 0].bar(labels, corr_u)
    axes[1, 0].set_title("Corr_u (higher better)")
    axes[1, 0].set_ylim(-1.0, 1.0)
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].bar(labels, corr_v)
    axes[1, 1].set_title("Corr_v (higher better)")
    axes[1, 1].set_ylim(-1.0, 1.0)
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.suptitle("KAN/MLP/CMEMS vs GDP_val")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_png}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recon-kan", default=r"D:\data\recon_kan.nc", help="KAN reconstruction netCDF")
    p.add_argument("--recon-mlp", default=r"D:\data\recon_mlp.nc", help="MLP reconstruction netCDF")
    p.add_argument("--cmems", default=r"D:\data\uswc_2023_cmems_cleaned.nc")
    p.add_argument("--gdp", default=r"D:\download\uswc_drifter_6hour_2023_corrected.nc")
    p.add_argument("--val-days", type=int, default=30)
    p.add_argument("--gdp-lon-col", default=None)
    p.add_argument("--gdp-lat-col", default=None)
    p.add_argument("--gdp-time-col", default=None)
    p.add_argument("--gdp-u-col", default=None)
    p.add_argument("--gdp-v-col", default=None)
    p.add_argument("--max-abs-speed", type=float, default=5.0, help="Filter points outside physical speed range in m/s")
    p.add_argument("--plot-out", default="compare_kan_mlp_vs_cmems.png", help="Save comparison figure PNG")
    args = p.parse_args()

    gdp = _load_gdp(
        args.gdp,
        lon_col=args.gdp_lon_col,
        lat_col=args.gdp_lat_col,
        time_col=args.gdp_time_col,
        u_col=args.gdp_u_col,
        v_col=args.gdp_v_col,
    )
    gdp_val = _split_val(gdp, args.val_days)
    print(f"GDP validation points: {len(gdp_val)}")

    ds_kan = _open_dataset_safe(args.recon_kan, label="KAN reconstruction")
    ds_mlp = _open_dataset_safe(args.recon_mlp, label="MLP reconstruction")
    ds_cmems = _open_dataset_safe(args.cmems, label="CMEMS")

    u_kan, v_kan, u_kan_units, v_kan_units = _interp_to_points(ds_kan, gdp_val)
    u_mlp, v_mlp, u_mlp_units, v_mlp_units = _interp_to_points(ds_mlp, gdp_val)
    u_cmems, v_cmems, u_cmems_units, v_cmems_units = _interp_to_points(ds_cmems, gdp_val)

    u_true = gdp_val["u"].to_numpy()
    v_true = gdp_val["v"].to_numpy()
    _brief_stats("RAW GDP", u_true, v_true)
    _brief_stats("RAW KAN", u_kan, v_kan)
    _brief_stats("RAW MLP", u_mlp, v_mlp)
    _brief_stats("RAW CMEMS", u_cmems, v_cmems)

    u_true, su_t = _scale_to_mps(u_true)
    v_true, sv_t = _scale_to_mps(v_true)
    u_kan, su_k = _scale_to_mps(u_kan, u_kan_units)
    v_kan, sv_k = _scale_to_mps(v_kan, v_kan_units)
    u_mlp, su_m = _scale_to_mps(u_mlp, u_mlp_units)
    v_mlp, sv_m = _scale_to_mps(v_mlp, v_mlp_units)
    u_cmems, su_c = _scale_to_mps(u_cmems, u_cmems_units)
    v_cmems, sv_c = _scale_to_mps(v_cmems, v_cmems_units)

    print(
        "Auto scale to m/s -> "
        f"GDP(u,v)=({su_t:.6g},{sv_t:.6g}), "
        f"KAN(u,v)=({su_k:.6g},{sv_k:.6g}), "
        f"MLP(u,v)=({su_m:.6g},{sv_m:.6g}), "
        f"CMEMS(u,v)=({su_c:.6g},{sv_c:.6g})"
    )
    _brief_stats("MPS GDP", u_true, v_true)
    _brief_stats("MPS KAN", u_kan, v_kan)
    _brief_stats("MPS MLP", u_mlp, v_mlp)
    _brief_stats("MPS CMEMS", u_cmems, v_cmems)

    u_true, v_true = _mask_unphysical(u_true, v_true, args.max_abs_speed, "GDP")
    u_kan, v_kan = _mask_unphysical(u_kan, v_kan, args.max_abs_speed, "KAN")
    u_mlp, v_mlp = _mask_unphysical(u_mlp, v_mlp, args.max_abs_speed, "MLP")
    u_cmems, v_cmems = _mask_unphysical(u_cmems, v_cmems, args.max_abs_speed, "CMEMS")

    m_kan = _metrics(u_kan, v_kan, u_true, v_true)
    m_mlp = _metrics(u_mlp, v_mlp, u_true, v_true)
    m_cmems = _metrics(u_cmems, v_cmems, u_true, v_true)

    _print("KAN Reconstruction vs GDP_val", m_kan)
    _print("MLP Reconstruction vs GDP_val", m_mlp)
    _print("CMEMS vs GDP_val", m_cmems)

    imp_vec_kan = (m_cmems.rmse_vec - m_kan.rmse_vec) / (m_cmems.rmse_vec + 1e-12) * 100.0
    imp_vec_mlp = (m_cmems.rmse_vec - m_mlp.rmse_vec) / (m_cmems.rmse_vec + 1e-12) * 100.0
    imp_spd_kan = (m_cmems.rmse_speed - m_kan.rmse_speed) / (m_cmems.rmse_speed + 1e-12) * 100.0
    imp_spd_mlp = (m_cmems.rmse_speed - m_mlp.rmse_speed) / (m_cmems.rmse_speed + 1e-12) * 100.0
    print(f"\nRMSE_vec improvement over CMEMS   | KAN={imp_vec_kan:.2f}% | MLP={imp_vec_mlp:.2f}%")
    print(f"RMSE_speed improvement over CMEMS | KAN={imp_spd_kan:.2f}% | MLP={imp_spd_mlp:.2f}%")

    better_vec = "KAN" if m_kan.rmse_vec < m_mlp.rmse_vec else "MLP"
    better_spd = "KAN" if m_kan.rmse_speed < m_mlp.rmse_speed else "MLP"
    print(f"Lower RMSE_vec among reconstructions: {better_vec}")
    print(f"Lower RMSE_speed among reconstructions: {better_spd}")
    _save_plot(m_kan, m_mlp, m_cmems, args.plot_out)


if __name__ == "__main__":
    main()
