"""Evaluate checkpoint-inferred flow (from data.nc + best_sr) against HF truth."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr

U_CANDS = ["u", "uo", "u_clean", "u_hr", "u_recon", "eastward_sea_water_velocity"]
V_CANDS = ["v", "vo", "v_clean", "v_hr", "v_recon", "northward_sea_water_velocity"]
LON_CANDS = ["lon", "longitude"]
LAT_CANDS = ["lat", "latitude"]
TIME_CANDS = ["time", "valid_time"]


class SmallSRNet(nn.Module):
    def __init__(self, scale: int = 2, ch: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(2, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(ch, 2 * (scale**2), 3, padding=1)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.tail(self.head(x)))


def _adapt_state_dict_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("body.") for k in state) and not any(k.startswith("head.") for k in state):
        return {("head." + k[len("body."):]) if k.startswith("body.") else k: v for k, v in state.items()}
    return state


def _find(ds: xr.Dataset, cands: list[str], label: str) -> str:
    for c in cands:
        if c in ds.data_vars:
            return c
    for c in cands:
        if c in ds.variables or c in ds.coords:
            return c
    raise KeyError(f"Cannot find {label}. candidates={cands}, available={list(ds.variables)}")


def _open(path: str | Path) -> xr.Dataset:
    try:
        return xr.open_dataset(path, mask_and_scale=False)
    except ValueError:
        return xr.open_dataset(path, engine="scipy", mask_and_scale=False)


def _spatial_stride(ds: xr.Dataset, factor: int) -> xr.Dataset:
    if factor <= 1:
        return ds
    lat_name = _find(ds, LAT_CANDS, "lat")
    lon_name = _find(ds, LON_CANDS, "lon")
    spatial_dims = []
    for d in ds[lat_name].dims + ds[lon_name].dims:
        if d not in spatial_dims:
            spatial_dims.append(d)
    indexers = {d: slice(None, None, factor) for d in spatial_dims}
    return ds.isel(indexers)


def _normalize_lon(lon: np.ndarray, target_lon: xr.DataArray) -> np.ndarray:
    mn, mx = float(target_lon.min()), float(target_lon.max())
    if mn >= 0 and mx > 180:
        return np.mod(lon, 360.0)
    return ((lon + 180.0) % 360.0) - 180.0


def _scale_to_mps(x: np.ndarray, units_hint: str = "") -> np.ndarray:
    u = units_hint.lower()
    if "cm/s" in u or "cm s-1" in u or "cm s^-1" in u:
        return x / 100.0
    if "m/day" in u or "m d-1" in u:
        return x / 86400.0
    if "km/day" in u or "km d-1" in u:
        return x * (1000.0 / 86400.0)
    return x


def _maybe_auto_unit_align(
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    u_true: np.ndarray,
    v_true: np.ndarray,
    pred_units_hint: str,
    true_units_hint: str,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not enabled:
        return u_pred, v_pred, 1.0
    pu = pred_units_hint.strip().lower()
    tu = true_units_hint.strip().lower()
    # only trigger when units are missing/ambiguous on prediction side
    if pu:
        return u_pred, v_pred, 1.0
    sp = np.sqrt(u_pred**2 + v_pred**2)
    st = np.sqrt(u_true**2 + v_true**2)
    sp = sp[np.isfinite(sp)]
    st = st[np.isfinite(st)]
    if sp.size == 0 or st.size == 0:
        return u_pred, v_pred, 1.0
    med_p = float(np.nanmedian(sp))
    med_t = float(np.nanmedian(st))
    if med_p <= 0 or med_t <= 0:
        return u_pred, v_pred, 1.0
    ratio = med_p / med_t
    factor = 1.0
    if 30.0 <= ratio <= 300.0:
        factor = 0.01
    elif 300.0 < ratio <= 3000.0:
        factor = 0.001
    elif 0.003 <= ratio <= 0.03:
        factor = 100.0
    elif 0.0003 <= ratio < 0.003:
        factor = 1000.0
    if factor != 1.0:
        print(f"[WARN] auto unit-align applied: factor={factor} (pred/true median speed ratio={ratio:.2f})")
        return u_pred * factor, v_pred * factor, factor
    return u_pred, v_pred, 1.0


def _to_numpy_streamed(da: xr.DataArray, chunk_len: int = 128) -> np.ndarray:
    if da.ndim == 0:
        return np.asarray(da.values)
    if da.ndim == 1:
        out = np.empty(da.shape, dtype=da.dtype)
        for s in range(0, da.shape[0], chunk_len):
            e = min(s + chunk_len, da.shape[0])
            out[s:e] = np.asarray(da.isel({da.dims[0]: slice(s, e)}).values)
        return out

    chunk_dim = da.dims[-2]
    out = np.empty(da.shape, dtype=da.dtype)
    for s in range(0, da.sizes[chunk_dim], chunk_len):
        e = min(s + chunk_len, da.sizes[chunk_dim])
        part = np.asarray(da.isel({chunk_dim: slice(s, e)}).values)
        idx = [slice(None)] * da.ndim
        idx[da.get_axis_num(chunk_dim)] = slice(s, e)
        out[tuple(idx)] = part
    return out


def _read_array_resilient(ds: xr.Dataset, var_name: str, chunk_len: int = 64) -> np.ndarray:
    da = ds[var_name]
    try:
        return da.to_numpy()
    except MemoryError:
        pass
    except Exception:
        pass

    try:
        return _to_numpy_streamed(da, chunk_len=chunk_len)
    except Exception:
        pass

    src = ds.encoding.get("source")
    if src:
        for eng in (None, "netcdf4", "h5netcdf", "scipy"):
            try:
                if eng is None:
                    ds2 = xr.open_dataset(src, mask_and_scale=False)
                else:
                    ds2 = xr.open_dataset(src, engine=eng, mask_and_scale=False)
                try:
                    if var_name not in ds2:
                        continue
                    try:
                        return ds2[var_name].to_numpy()
                    except Exception:
                        return _to_numpy_streamed(ds2[var_name], chunk_len=max(8, chunk_len // 2))
                finally:
                    ds2.close()
            except Exception:
                continue
        # final fallback: read directly via netCDF4 with mask/scale disabled
        try:
            from netCDF4 import Dataset as NcDataset

            with NcDataset(src, mode="r") as nc:
                vname = var_name
                if vname not in nc.variables:
                    cands = U_CANDS + V_CANDS
                    for c in cands:
                        if c in nc.variables:
                            vname = c
                            break
                if vname not in nc.variables:
                    raise KeyError(f"Variable '{var_name}' not found in netCDF file: {src}")
                v = nc.variables[vname]
                try:
                    v.set_auto_maskandscale(False)
                except Exception:
                    pass
                shape = tuple(int(s) for s in v.shape)
                out = np.empty(shape, dtype=v.dtype)
                if len(shape) <= 1:
                    out[...] = v[:]
                    return out
                chunk_axis = max(0, len(shape) - 2)
                for s in range(0, shape[chunk_axis], chunk_len):
                    e = min(s + chunk_len, shape[chunk_axis])
                    key = [slice(None)] * len(shape)
                    key[chunk_axis] = slice(s, e)
                    out[tuple(key)] = v[tuple(key)]
                return out
        except Exception:
            pass
    raise RuntimeError(f"Failed to read variable '{var_name}' robustly from dataset")


def _metrics(u_pred: np.ndarray, v_pred: np.ndarray, u_true: np.ndarray, v_true: np.ndarray, max_abs_speed: float) -> dict[str, float]:
    m = (
        np.isfinite(u_pred)
        & np.isfinite(v_pred)
        & np.isfinite(u_true)
        & np.isfinite(v_true)
        & (np.abs(u_pred) <= max_abs_speed)
        & (np.abs(v_pred) <= max_abs_speed)
        & (np.abs(u_true) <= max_abs_speed)
        & (np.abs(v_true) <= max_abs_speed)
    )
    up, vp, ut, vt = u_pred[m], v_pred[m], u_true[m], v_true[m]
    if up.size == 0:
        return {"n": 0, "rmse_u": np.nan, "rmse_v": np.nan, "rmse_vec": np.nan, "rmse_speed": np.nan, "corr_u": np.nan, "corr_v": np.nan}

    sp = np.sqrt(up**2 + vp**2)
    st = np.sqrt(ut**2 + vt**2)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a, b)[0, 1]) if a.size > 2 else np.nan

    return {
        "n": int(up.size),
        "rmse_u": float(np.sqrt(np.mean((up - ut) ** 2))),
        "rmse_v": float(np.sqrt(np.mean((vp - vt) ** 2))),
        "rmse_vec": float(np.sqrt(np.mean((up - ut) ** 2 + (vp - vt) ** 2))),
        "rmse_speed": float(np.sqrt(np.mean((sp - st) ** 2))),
        "corr_u": _corr(up, ut),
        "corr_v": _corr(vp, vt),
    }


def _init_running_stats() -> dict[str, float]:
    return {
        "n": 0.0,
        "sum_du2": 0.0,
        "sum_dv2": 0.0,
        "sum_dvec2": 0.0,
        "sum_dspeed2": 0.0,
        "sum_up": 0.0,
        "sum_ut": 0.0,
        "sum_up2": 0.0,
        "sum_ut2": 0.0,
        "sum_uput": 0.0,
        "sum_vp": 0.0,
        "sum_vt": 0.0,
        "sum_vp2": 0.0,
        "sum_vt2": 0.0,
        "sum_vpvt": 0.0,
        "sum_sp": 0.0,
        "sum_st": 0.0,
    }


def _accumulate_metrics(stats: dict[str, float], u_pred: np.ndarray, v_pred: np.ndarray, u_true: np.ndarray, v_true: np.ndarray, max_abs_speed: float) -> None:
    m = (
        np.isfinite(u_pred)
        & np.isfinite(v_pred)
        & np.isfinite(u_true)
        & np.isfinite(v_true)
        & (np.abs(u_pred) <= max_abs_speed)
        & (np.abs(v_pred) <= max_abs_speed)
        & (np.abs(u_true) <= max_abs_speed)
        & (np.abs(v_true) <= max_abs_speed)
    )
    up = u_pred[m]
    vp = v_pred[m]
    ut = u_true[m]
    vt = v_true[m]
    _accumulate_metrics_from_values(stats, up, vp, ut, vt)


def _accumulate_metrics_from_values(stats: dict[str, float], up: np.ndarray, vp: np.ndarray, ut: np.ndarray, vt: np.ndarray) -> None:
    if up.size == 0:
        return
    sp = np.sqrt(up**2 + vp**2)
    st = np.sqrt(ut**2 + vt**2)
    stats["n"] += float(up.size)
    stats["sum_du2"] += float(np.sum((up - ut) ** 2))
    stats["sum_dv2"] += float(np.sum((vp - vt) ** 2))
    stats["sum_dvec2"] += float(np.sum((up - ut) ** 2 + (vp - vt) ** 2))
    stats["sum_dspeed2"] += float(np.sum((sp - st) ** 2))
    stats["sum_up"] += float(np.sum(up))
    stats["sum_ut"] += float(np.sum(ut))
    stats["sum_up2"] += float(np.sum(up**2))
    stats["sum_ut2"] += float(np.sum(ut**2))
    stats["sum_uput"] += float(np.sum(up * ut))
    stats["sum_vp"] += float(np.sum(vp))
    stats["sum_vt"] += float(np.sum(vt))
    stats["sum_vp2"] += float(np.sum(vp**2))
    stats["sum_vt2"] += float(np.sum(vt**2))
    stats["sum_vpvt"] += float(np.sum(vp * vt))
    stats["sum_sp"] += float(np.sum(sp))
    stats["sum_st"] += float(np.sum(st))


def _finalize_metrics(stats: dict[str, float]) -> dict[str, float]:
    n = int(stats["n"])
    if n == 0:
        return {
            "n": 0,
            "rmse_u": np.nan,
            "rmse_v": np.nan,
            "rmse_vec": np.nan,
            "rmse_speed": np.nan,
            "corr_u": np.nan,
            "corr_v": np.nan,
            "mean_speed_pred": np.nan,
            "mean_speed_true": np.nan,
        }

    def _corr(n_: int, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float:
        num = n_ * sxy - sx * sy
        den_term = (n_ * sxx - sx * sx) * (n_ * syy - sy * sy)
        if den_term <= 1e-12:
            return np.nan
        return float(num / np.sqrt(den_term))

    return {
        "n": n,
        "rmse_u": float(np.sqrt(stats["sum_du2"] / n)),
        "rmse_v": float(np.sqrt(stats["sum_dv2"] / n)),
        "rmse_vec": float(np.sqrt(stats["sum_dvec2"] / n)),
        "rmse_speed": float(np.sqrt(stats["sum_dspeed2"] / n)),
        "corr_u": _corr(n, stats["sum_up"], stats["sum_ut"], stats["sum_up2"], stats["sum_ut2"], stats["sum_uput"]),
        "corr_v": _corr(n, stats["sum_vp"], stats["sum_vt"], stats["sum_vp2"], stats["sum_vt2"], stats["sum_vpvt"]),
        "mean_speed_pred": float(stats["sum_sp"] / n),
        "mean_speed_true": float(stats["sum_st"] / n),
    }


def _manual_nearest_sample(src_t: xr.Dataset, u_name: str, v_name: str, lon_name: str, lat_name: str, lon_q: np.ndarray, lat_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon_src = src_t[lon_name].values
    lat_src = src_t[lat_name].values
    q_lon = lon_q.ravel()
    q_lat = lat_q.ravel()

    if lon_src.ndim == 1 and lat_src.ndim == 1:
        lon_arr = lon_src
        lat_arr = lat_src
        lon_rev = lon_arr[0] > lon_arr[-1]
        lat_rev = lat_arr[0] > lat_arr[-1]
        if lon_rev:
            lon_arr = lon_arr[::-1]
        if lat_rev:
            lat_arr = lat_arr[::-1]

        def nearest_idx(arr: np.ndarray, vals: np.ndarray) -> np.ndarray:
            out = np.empty(vals.shape[0], dtype=np.int32)
            hi = len(arr) - 1
            for i, vv in enumerate(vals):
                j = int(np.searchsorted(arr, vv))
                if j < 1:
                    j = 1
                elif j > hi:
                    j = hi
                left = arr[j - 1]
                right = arr[j]
                out[i] = j - 1 if abs(vv - left) <= abs(vv - right) else j
            return out

        xi = nearest_idx(lon_arr, q_lon)
        yi = nearest_idx(lat_arr, q_lat)
        if lon_rev:
            xi = (len(lon_src) - 1) - xi
        if lat_rev:
            yi = (len(lat_src) - 1) - yi
        u_np = src_t[u_name].values[yi, xi].reshape(lon_q.shape)
        v_np = src_t[v_name].values[yi, xi].reshape(lon_q.shape)
        return u_np, v_np

    pts_src = np.column_stack([lon_src.ravel(), lat_src.ravel()])
    q_pts = np.column_stack([q_lon, q_lat])
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(pts_src).query(q_pts, k=1)
    except Exception:  # noqa: BLE001
        idx = np.empty(q_pts.shape[0], dtype=np.int64)
        for i, (qlon, qlat) in enumerate(q_pts):
            d2 = (pts_src[:, 0] - qlon) ** 2 + (pts_src[:, 1] - qlat) ** 2
            idx[i] = int(np.argmin(d2))
    u_flat = src_t[u_name].values.ravel()
    v_flat = src_t[v_name].values.ravel()
    return u_flat[idx].reshape(lon_q.shape), v_flat[idx].reshape(lon_q.shape)


def _upscale_coord_1d(coord: np.ndarray, scale: int) -> np.ndarray:
    c = np.asarray(coord)
    if c.ndim != 1:
        raise ValueError("Only 1D coordinates are supported for upscale helper")
    n = c.shape[0]
    if n <= 1:
        return np.repeat(c, scale)
    start = float(c[0])
    end = float(c[-1])
    return np.linspace(start, end, n * scale, dtype=np.float64)


def _infer_bg_scale_factor(u_bg: np.ndarray, v_bg: np.ndarray) -> float:
    bg_abs = np.abs(np.concatenate([u_bg[np.isfinite(u_bg)], v_bg[np.isfinite(v_bg)]]))
    med = float(np.nanmedian(bg_abs)) if bg_abs.size else 0.0
    if med > 20.0:
        return 1000.0
    if med > 2.0:
        return 100.0
    return 1.0


def _manual_nearest_sample_rect(
    src_t: xr.Dataset, u_name: str, v_name: str, lon_name: str, lat_name: str, lon_q_1d: np.ndarray, lat_q_1d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lon_src = src_t[lon_name].values
    lat_src = src_t[lat_name].values
    if lon_src.ndim != 1 or lat_src.ndim != 1:
        lon_q, lat_q = np.meshgrid(lon_q_1d, lat_q_1d)
        return _manual_nearest_sample(src_t, u_name, v_name, lon_name, lat_name, lon_q, lat_q)

    lon_arr = lon_src
    lat_arr = lat_src
    lon_rev = lon_arr[0] > lon_arr[-1]
    lat_rev = lat_arr[0] > lat_arr[-1]
    if lon_rev:
        lon_arr = lon_arr[::-1]
    if lat_rev:
        lat_arr = lat_arr[::-1]

    def nearest_idx_1d(arr: np.ndarray, vals: np.ndarray) -> np.ndarray:
        out = np.empty(vals.shape[0], dtype=np.int32)
        hi = len(arr) - 1
        for i, vv in enumerate(vals):
            j = int(np.searchsorted(arr, vv))
            if j < 1:
                j = 1
            elif j > hi:
                j = hi
            left = arr[j - 1]
            right = arr[j]
            out[i] = j - 1 if abs(vv - left) <= abs(vv - right) else j
        return out

    xi = nearest_idx_1d(lon_arr, np.asarray(lon_q_1d).ravel())
    yi = nearest_idx_1d(lat_arr, np.asarray(lat_q_1d).ravel())
    if lon_rev:
        xi = (len(lon_src) - 1) - xi
    if lat_rev:
        yi = (len(lat_src) - 1) - yi

    u_src = src_t[u_name].values
    v_src = src_t[v_name].values
    return u_src[np.ix_(yi, xi)], v_src[np.ix_(yi, xi)]


def _load_bg_from_data(data_path: str) -> xr.Dataset:
    ds = _open(data_path)
    t_name = _find(ds, TIME_CANDS, "time")
    lat_name = _find(ds, LAT_CANDS, "lat")
    lon_name = _find(ds, LON_CANDS, "lon")
    u_bg = ds["u_bg"].transpose(t_name, lat_name, lon_name).values.astype(np.float32)
    v_bg = ds["v_bg"].transpose(t_name, lat_name, lon_name).values.astype(np.float32)
    ds.close()
    return xr.Dataset(
        data_vars={
            "u_bg": ((t_name, lat_name, lon_name), u_bg),
            "v_bg": ((t_name, lat_name, lon_name), v_bg),
        },
        coords={t_name: ds[t_name].values, lat_name: ds[lat_name].values, lon_name: ds[lon_name].values},
    )


def _infer_from_data_nc(args: argparse.Namespace) -> tuple[xr.Dataset, xr.Dataset]:
    ds = _open(args.data)
    t_name = _find(ds, TIME_CANDS, "time")
    lat_name = _find(ds, LAT_CANDS, "lat")
    lon_name = _find(ds, LON_CANDS, "lon")
    u_bg = ds["u_bg"].transpose(t_name, lat_name, lon_name).values.astype(np.float32)
    v_bg = ds["v_bg"].transpose(t_name, lat_name, lon_name).values.astype(np.float32)
    bg_scale = _infer_bg_scale_factor(u_bg, v_bg)
    u_in = u_bg / bg_scale
    v_in = v_bg / bg_scale

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = SmallSRNet(scale=args.scale, ch=args.channels).to(device)
    state = ckpt.get("model", ckpt.get("model_state_dict"))
    if state is None:
        raise KeyError("Checkpoint missing model/model_state_dict")
    model.load_state_dict(_adapt_state_dict_keys(state), strict=True)
    model.eval()

    if args.eval_grid_mode == "pooled":
        u_rec = np.zeros_like(u_bg, dtype=np.float32)
        v_rec = np.zeros_like(v_bg, dtype=np.float32)
    else:
        u_rec = np.zeros((u_bg.shape[0], u_bg.shape[1] * args.scale, u_bg.shape[2] * args.scale), dtype=np.float32)
        v_rec = np.zeros((v_bg.shape[0], v_bg.shape[1] * args.scale, v_bg.shape[2] * args.scale), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, u_bg.shape[0], args.batch_size):
            j = min(i + args.batch_size, u_bg.shape[0])
            x_t = torch.from_numpy(np.stack([u_in[i:j], v_in[i:j]], axis=1)).to(device)
            y_hr = model(x_t)
            if args.eval_grid_mode == "pooled":
                u_rec[i:j] = (F.avg_pool2d(y_hr[:, 0:1], kernel_size=args.scale, stride=args.scale).squeeze(1).cpu().numpy() * bg_scale)
                v_rec[i:j] = (F.avg_pool2d(y_hr[:, 1:2], kernel_size=args.scale, stride=args.scale).squeeze(1).cpu().numpy() * bg_scale)
            else:
                u_rec[i:j] = y_hr[:, 0].cpu().numpy() * bg_scale
                v_rec[i:j] = y_hr[:, 1].cpu().numpy() * bg_scale

    bg_ds = _load_bg_from_data(args.data)
    if args.eval_grid_mode == "pooled":
        recon_ds = xr.Dataset(
            data_vars={"u_recon": ((t_name, lat_name, lon_name), u_rec), "v_recon": ((t_name, lat_name, lon_name), v_rec)},
            coords={t_name: ds[t_name].values, lat_name: ds[lat_name].values, lon_name: ds[lon_name].values},
        )
    else:
        lat_hr = _upscale_coord_1d(ds[lat_name].values, args.scale)
        lon_hr = _upscale_coord_1d(ds[lon_name].values, args.scale)
        recon_ds = xr.Dataset(
            data_vars={"u_recon": ((t_name, lat_name, lon_name), u_rec), "v_recon": ((t_name, lat_name, lon_name), v_rec)},
            coords={t_name: ds[t_name].values, lat_name: lat_hr, lon_name: lon_hr},
        )
    return recon_ds, bg_ds


def _interp_to_hf(
    src: xr.Dataset,
    hf: xr.Dataset,
    u_var: str,
    v_var: str,
    interp_method: str = "auto",
    max_time_gap_hours: float | None = None,
    auto_unit_align: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    slon = _find(src, LON_CANDS, "lon")
    slat = _find(src, LAT_CANDS, "lat")
    stime = _find(src, TIME_CANDS, "time")

    hu = _find(hf, U_CANDS, "u")
    hv = _find(hf, V_CANDS, "v")
    hlon = _find(hf, LON_CANDS, "lon")
    hlat = _find(hf, LAT_CANDS, "lat")
    htime = _find(hf, TIME_CANDS, "time")

    lon_hf = _normalize_lon(hf[hlon].values, src[slon])
    t_time = hf[htime].values.astype("datetime64[ns]")
    if np.size(t_time) == 1:
        tt = t_time.reshape(-1)[0]
        src_t = src.sel({stime: tt}, method="nearest")
        if max_time_gap_hours is not None:
            src_tt = np.asarray(src_t[stime].values).reshape(-1)[0].astype("datetime64[ns]")
            gap_h = abs((src_tt - tt) / np.timedelta64(1, "h"))
            if float(gap_h) > float(max_time_gap_hours):
                raise RuntimeError(f"time gap too large: {float(gap_h):.2f}h > {max_time_gap_hours}h")
        out = None
        if interp_method in {"auto", "linear"}:
            try:
                out = src_t.interp(
                    {
                        slon: xr.DataArray(lon_hf, dims=hf[hlon].dims),
                        slat: xr.DataArray(hf[hlat].values, dims=hf[hlat].dims),
                    },
                    method="linear",
                )
            except Exception as exc:  # noqa: BLE001
                if interp_method == "linear":
                    raise
                if "Unable to allocate" not in str(exc):
                    raise
                print("[WARN] spatial linear interp OOM -> fallback to nearest")
        if out is None:
            lat_hf = hf[hlat].values
            if lon_hf.ndim == 1 and lat_hf.ndim == 1:
                dims_q = (hf[hlat].dims[0], hf[hlon].dims[0])
                u_np, v_np = _manual_nearest_sample_rect(src_t, u_var, v_var, slon, slat, lon_hf, lat_hf)
            else:
                lon_q, lat_q = lon_hf, lat_hf
                dims_q = hf[hlon].dims
                u_np, v_np = _manual_nearest_sample(src_t, u_var, v_var, slon, slat, lon_q, lat_q)
            out = xr.Dataset({u_var: (dims_q, u_np), v_var: (dims_q, v_np)})
    else:
        chunks: list[xr.Dataset] = []
        for tt in np.ravel(t_time):
            src_t = src.sel({stime: tt}, method="nearest")
            if max_time_gap_hours is not None:
                src_tt = np.asarray(src_t[stime].values).reshape(-1)[0].astype("datetime64[ns]")
                gap_h = abs((src_tt - tt) / np.timedelta64(1, "h"))
                if float(gap_h) > float(max_time_gap_hours):
                    raise RuntimeError(f"time gap too large: {float(gap_h):.2f}h > {max_time_gap_hours}h")
            part = None
            if interp_method in {"auto", "linear"}:
                try:
                    part = src_t.interp(
                        {
                            slon: xr.DataArray(lon_hf, dims=hf[hlon].dims),
                            slat: xr.DataArray(hf[hlat].values, dims=hf[hlat].dims),
                        },
                        method="linear",
                    )
                except Exception as exc:  # noqa: BLE001
                    if interp_method == "linear":
                        raise
                    if "Unable to allocate" not in str(exc):
                        raise
                    print("[WARN] spatial linear interp OOM -> fallback to nearest")
            if part is None:
                lat_hf = hf[hlat].values
                if lon_hf.ndim == 1 and lat_hf.ndim == 1:
                    dims_q = (hf[hlat].dims[0], hf[hlon].dims[0])
                    u_np, v_np = _manual_nearest_sample_rect(src_t, u_var, v_var, slon, slat, lon_hf, lat_hf)
                else:
                    lon_q, lat_q = lon_hf, lat_hf
                    dims_q = hf[hlon].dims
                    u_np, v_np = _manual_nearest_sample(src_t, u_var, v_var, slon, slat, lon_q, lat_q)
                part = xr.Dataset({u_var: (dims_q, u_np), v_var: (dims_q, v_np)})
            part = part.expand_dims({stime: [tt]})
            chunks.append(part)
        out = xr.concat(chunks, dim=stime)

    out_ds = out[[u_var, v_var]]
    u_pred_np = _read_array_resilient(out_ds, u_var, chunk_len=64)
    v_pred_np = _read_array_resilient(out_ds, v_var, chunk_len=64)
    u_true_np = _read_array_resilient(hf, hu, chunk_len=64)
    v_true_np = _read_array_resilient(hf, hv, chunk_len=64)

    pred_units = str(src[u_var].attrs.get("units", "")) + "|" + str(src[v_var].attrs.get("units", ""))
    true_units = str(hf[hu].attrs.get("units", "")) + "|" + str(hf[hv].attrs.get("units", ""))
    u_pred = _scale_to_mps(u_pred_np, pred_units)
    v_pred = _scale_to_mps(v_pred_np, pred_units)
    u_true = _scale_to_mps(u_true_np, true_units)
    v_true = _scale_to_mps(v_true_np, true_units)
    u_pred, v_pred, _ = _maybe_auto_unit_align(u_pred, v_pred, u_true, v_true, pred_units, true_units, enabled=auto_unit_align)
    return u_pred, v_pred, u_true, v_true


def _interp_to_hf_chunked(
    src: xr.Dataset,
    hf: xr.Dataset,
    u_var: str,
    v_var: str,
    interp_method: str = "auto",
    lat_chunk: int = 128,
    max_time_gap_hours: float | None = None,
    auto_unit_align: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hlat = _find(hf, LAT_CANDS, "lat")
    lat_dims = hf[hlat].dims
    if len(lat_dims) != 1:
        return _interp_to_hf(src, hf, u_var, v_var, interp_method=interp_method)
    lat_dim = lat_dims[0]
    nlat = hf.sizes[lat_dim]
    up_all: list[np.ndarray] = []
    vp_all: list[np.ndarray] = []
    ut_all: list[np.ndarray] = []
    vt_all: list[np.ndarray] = []
    for s in range(0, nlat, lat_chunk):
        e = min(s + lat_chunk, nlat)
        hf_chunk = hf.isel({lat_dim: slice(s, e)})
        up, vp, ut, vt = _interp_to_hf(
            src,
            hf_chunk,
            u_var,
            v_var,
            interp_method=interp_method,
            max_time_gap_hours=max_time_gap_hours,
            auto_unit_align=auto_unit_align,
        )
        if up.ndim == 3:
            up, vp = up[0], vp[0]
        if ut.ndim == 3:
            ut, vt = ut[0], vt[0]
        up_all.append(up.ravel())
        vp_all.append(vp.ravel())
        ut_all.append(ut.ravel())
        vt_all.append(vt.ravel())
    return np.concatenate(up_all), np.concatenate(vp_all), np.concatenate(ut_all), np.concatenate(vt_all)


def _collect_hf_time_range(hf_files: list[Path]) -> tuple[np.datetime64, np.datetime64] | None:
    tmin = None
    tmax = None
    for fp in hf_files:
        try:
            ds = _open(fp)
            try:
                tname = _find(ds, TIME_CANDS, "time")
                tv = np.asarray(ds[tname].values).astype("datetime64[ns]").ravel()
                if tv.size == 0:
                    continue
                tt0 = tv.min()
                tt1 = tv.max()
                tmin = tt0 if tmin is None else min(tmin, tt0)
                tmax = tt1 if tmax is None else max(tmax, tt1)
            finally:
                ds.close()
        except Exception:
            continue
    if tmin is None or tmax is None:
        return None
    return tmin, tmax


def _slice_src_to_hf_time(src: xr.Dataset, pad_hours: int, hf_time_range: tuple[np.datetime64, np.datetime64] | None) -> xr.Dataset:
    if hf_time_range is None:
        return src
    stime = _find(src, TIME_CANDS, "time")
    t0, t1 = hf_time_range
    pad = np.timedelta64(int(max(0, pad_hours)), "h")
    w0 = t0 - pad
    w1 = t1 + pad
    try:
        src_s = src.sel({stime: slice(w0, w1)})
        if src_s[stime].size > 0:
            return src_s
    except Exception:
        pass
    return src


def main(args: argparse.Namespace) -> None:
    if args.recon:
        recon_ds = _open(args.recon)
        bg_ds = _load_bg_from_data(args.data)
        print(f"[INFO] Using pre-exported reconstruction for evaluation: {args.recon}")
    else:
        recon_ds, bg_ds = _infer_from_data_nc(args)
    hf_files = sorted(Path(args.hf_dir).glob("*.nc"))
    if not hf_files:
        raise FileNotFoundError(f"No HF .nc files in {args.hf_dir}")
    hf_time_range = _collect_hf_time_range(hf_files)
    recon_ds = _slice_src_to_hf_time(recon_ds, args.hf_time_pad_hours, hf_time_range)
    bg_ds = _slice_src_to_hf_time(bg_ds, args.hf_time_pad_hours, hf_time_range)

    specs = [("PINN-SR", recon_ds, "u_recon", "v_recon"), ("CMEMS-bg", bg_ds, "u_bg", "v_bg")]
    stats_map = {name: _init_running_stats() for name, *_ in specs}
    file_used = 0
    file_skipped = 0

    for f in hf_files:
        hf = _open(f)
        try:
            file_res: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
            for name, src_ds, uvar, vvar in specs:
                up, vp, ut, vt = _interp_to_hf(
                    src_ds,
                    hf,
                    uvar,
                    vvar,
                    interp_method=args.interp_method,
                    max_time_gap_hours=args.max_time_gap_hours,
                    auto_unit_align=args.auto_unit_align,
                )
                if up.ndim == 3:
                    up, vp = up[0], vp[0]
                if ut.ndim == 3:
                    ut, vt = ut[0], vt[0]
                file_res[name] = (up, vp, ut, vt)
        except Exception as exc:  # noqa: BLE001
            file_skipped += 1
            print(f"[WARN] skip HF file due to eval failure: {f.name} | {exc}")
            continue
        finally:
            hf.close()
        if not file_res:
            file_skipped += 1
            continue
        file_used += 1
        up_sr, vp_sr, ut_sr, vt_sr = file_res["PINN-SR"]
        up_bg, vp_bg, ut_bg, vt_bg = file_res["CMEMS-bg"]
        up_sr = up_sr.ravel(); vp_sr = vp_sr.ravel(); ut_sr = ut_sr.ravel(); vt_sr = vt_sr.ravel()
        up_bg = up_bg.ravel(); vp_bg = vp_bg.ravel(); ut_bg = ut_bg.ravel(); vt_bg = vt_bg.ravel()
        # Fair comparison: evaluate both models on exactly the same valid samples.
        common = (
            np.isfinite(ut_sr)
            & np.isfinite(vt_sr)
            & np.isfinite(up_sr)
            & np.isfinite(vp_sr)
            & np.isfinite(up_bg)
            & np.isfinite(vp_bg)
            & (np.abs(ut_sr) <= args.max_abs_speed)
            & (np.abs(vt_sr) <= args.max_abs_speed)
            & (np.abs(up_sr) <= args.max_abs_speed)
            & (np.abs(vp_sr) <= args.max_abs_speed)
            & (np.abs(up_bg) <= args.max_abs_speed)
            & (np.abs(vp_bg) <= args.max_abs_speed)
        )
        _accumulate_metrics_from_values(stats_map["PINN-SR"], up_sr[common], vp_sr[common], ut_sr[common], vt_sr[common])
        _accumulate_metrics_from_values(stats_map["CMEMS-bg"], up_bg[common], vp_bg[common], ut_bg[common], vt_bg[common])

    print(f"[INFO] common-file evaluation used={file_used}, skipped={file_skipped}, total={len(hf_files)}")
    rows = []
    for name, *_ in specs:
        if int(stats_map[name]["n"]) == 0:
            raise RuntimeError(f"No valid samples for model={name} on common-file evaluation set.")
        m = _finalize_metrics(stats_map[name])
        m["model"] = name
        rows.append(m)
    if len(rows) == 2:
        by_name = {r["model"]: r for r in rows}
        if "PINN-SR" in by_name and "CMEMS-bg" in by_name:
            base = by_name["CMEMS-bg"]["rmse_vec"]
            gain = (base - by_name["PINN-SR"]["rmse_vec"]) / (base + 1e-12) * 100.0
            by_name["PINN-SR"]["rmse_vec_gain_vs_cmems_pct"] = gain
            by_name["CMEMS-bg"]["rmse_vec_gain_vs_cmems_pct"] = 0.0
            rows = [by_name["PINN-SR"], by_name["CMEMS-bg"]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "hf_eval_from_data_ckpt.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[INFO] Saved: {out_csv}")
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate data.nc + best_sr inferred flow against HF truth")
    p.add_argument("--data", default=r"D:/data/output/train_data_2023.nc")
    p.add_argument("--ckpt", default=r"D:/data/output/checkpoints_sr_mlp_strict_weighted/best_sr.pt")
    p.add_argument("--recon", default="", help="optional pre-exported recon nc (u_recon/v_recon). If set, skip on-the-fly checkpoint inference.")
    p.add_argument("--hf-dir", default=r"D:/生产力/py程序/pythonProject/hf/202307_uswc_1km_rtv_sio")
    p.add_argument("--out-dir", default=r"D:/data/output/hf_eval")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-abs-speed", type=float, default=5.0)
    p.add_argument("--max-time-gap-hours", type=float, default=3.0, help="max allowed nearest-time gap between source and HF timestamps")
    p.add_argument("--auto-unit-align", action=argparse.BooleanOptionalAction, default=True, help="auto-correct unit scale when prediction units metadata is missing")
    p.add_argument("--interp-method", choices=["auto", "linear", "nearest"], default="auto")
    p.add_argument("--eval-grid-mode", choices=["native_hr", "pooled"], default="native_hr", help="native_hr keeps SR output grid; pooled matches LR via avg-pool")
    p.add_argument("--hf-time-pad-hours", type=int, default=12, help="time-window pad around HF period when slicing source dataset")
    p.add_argument("--cpu", action="store_true")
    main(p.parse_args())
