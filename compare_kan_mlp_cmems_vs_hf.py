"""Compare KAN/MLP/CMEMS against HF-radar reference truth."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    corr_u: float
    corr_v: float


@dataclass
class RunningStats:
    n: int = 0
    sum_du2: float = 0.0
    sum_dv2: float = 0.0
    sum_dvec2: float = 0.0
    sum_dspeed2: float = 0.0
    sum_up: float = 0.0
    sum_ut: float = 0.0
    sum_up2: float = 0.0
    sum_ut2: float = 0.0
    sum_uput: float = 0.0
    sum_vp: float = 0.0
    sum_vt: float = 0.0
    sum_vp2: float = 0.0
    sum_vt2: float = 0.0
    sum_vpvt: float = 0.0


def _find(ds: xr.Dataset, cands: list[str], label: str) -> str:
    for c in cands:
        if c in ds.variables or c in ds.coords:
            return c
    raise KeyError(f"Cannot find {label}. candidates={cands}, available={list(ds.variables)}")


def _open_dataset_safe(path: str | Path, label: str) -> xr.Dataset:
    try:
        return xr.open_dataset(path, mask_and_scale=False)
    except ValueError as exc:
        msg = str(exc)
        if "their dependencies may not be installed" in msg:
            try:
                return xr.open_dataset(path, engine="scipy", mask_and_scale=False)
            except Exception as scipy_exc:  # noqa: BLE001
                raise RuntimeError(
                    f"无法打开 {label} 文件: {path}\n"
                    "请安装 netCDF4/h5netcdf/scipy，例如: pip install netCDF4\n"
                    f"原始错误: {exc}\nscipy重试错误: {scipy_exc}"
                ) from scipy_exc
        raise


def _normalize_lon(lon: np.ndarray, target_lon: xr.DataArray) -> np.ndarray:
    mn, mx = float(target_lon.min()), float(target_lon.max())
    if mn >= 0 and mx > 180:
        return np.mod(lon, 360.0)
    return ((lon + 180.0) % 360.0) - 180.0


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
    # Unknown units: keep data unchanged to avoid accidental auto-rescaling.
    # (Heuristic guessing can silently amplify/reduce flow by 100x.)
    return x, 1.0


def _list_hf_files(hf_dir: str) -> list[Path]:
    files = sorted(Path(hf_dir).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found under {hf_dir}")
    return files


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


def _to_numpy_streamed(da: xr.DataArray, chunk_len: int = 128) -> np.ndarray:
    """Read DataArray in chunks to reduce peak backend read memory."""
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


def _manual_nearest_sample(src_t: xr.Dataset, u_name: str, v_name: str, lon_name: str, lat_name: str, lon_q: np.ndarray, lat_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon_src = src_t[lon_name].values
    lat_src = src_t[lat_name].values
    q_lon = lon_q.ravel()
    q_lat = lat_q.ravel()

    if lon_src.ndim == 1 and lat_src.ndim == 1:
        # ensure ascending for searchsorted
        lon_arr = lon_src
        lat_arr = lat_src
        lon_rev = lon_arr[0] > lon_arr[-1]
        lat_rev = lat_arr[0] > lat_arr[-1]
        if lon_rev:
            lon_arr = lon_arr[::-1]
        if lat_rev:
            lat_arr = lat_arr[::-1]

        def nearest_idx(arr: np.ndarray, vals: np.ndarray) -> np.ndarray:
            # Scalar loop avoids large temporary arrays in ultra-low-memory environments.
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

    # curvilinear lon/lat fallback
    pts_src = np.column_stack([lon_src.ravel(), lat_src.ravel()])
    q_pts = np.column_stack([q_lon, q_lat])
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(pts_src).query(q_pts, k=1)
    except Exception:  # noqa: BLE001
        # scipy may be unavailable in minimal environments;
        # keep a pure-numpy fallback for robustness.
        idx = np.empty(q_pts.shape[0], dtype=np.int64)
        for i, (qlon, qlat) in enumerate(q_pts):
            d2 = (pts_src[:, 0] - qlon) ** 2 + (pts_src[:, 1] - qlat) ** 2
            idx[i] = int(np.argmin(d2))
    u_flat = src_t[u_name].values.ravel()
    v_flat = src_t[v_name].values.ravel()
    return u_flat[idx].reshape(lon_q.shape), v_flat[idx].reshape(lon_q.shape)


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


def _to_common_grid(src: xr.Dataset, ref: xr.Dataset, interp_method: str = "auto") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    su = _find(src, U_CANDS, "u")
    sv = _find(src, V_CANDS, "v")
    slon = _find(src, LON_CANDS, "lon")
    slat = _find(src, LAT_CANDS, "lat")
    stime = _find(src, TIME_CANDS, "time")

    ru = _find(ref, U_CANDS, "u")
    rv = _find(ref, V_CANDS, "v")
    rlon = _find(ref, LON_CANDS, "lon")
    rlat = _find(ref, LAT_CANDS, "lat")
    rtime = _find(ref, TIME_CANDS, "time")

    t_lon = ref[rlon].values
    t_lat = ref[rlat].values
    t_time = ref[rtime].values.astype("datetime64[ns]")
    t_lon = _normalize_lon(t_lon, src[slon])

    # memory-safe interpolation:
    # 1) pick nearest time slice first (HF files are typically single-timestep)
    # 2) then do spatial interp only
    if np.size(t_time) == 1:
        src_t = src.sel({stime: t_time.reshape(-1)[0]}, method="nearest")
        src_i = None
        if interp_method in {"auto", "linear"}:
            try:
                src_i = src_t.interp(
                    {
                        slon: xr.DataArray(t_lon, dims=ref[rlon].dims),
                        slat: xr.DataArray(t_lat, dims=ref[rlat].dims),
                    },
                    method="linear",
                )
            except Exception as exc:  # noqa: BLE001
                if interp_method == "linear":
                    raise
                if "Unable to allocate" not in str(exc):
                    raise
                print("[WARN] spatial linear interp OOM -> fallback to nearest")
        if src_i is None:
            if t_lon.ndim == 1 and t_lat.ndim == 1:
                dims_q = (ref[rlat].dims[0], ref[rlon].dims[0])
                u_np, v_np = _manual_nearest_sample_rect(src_t, su, sv, slon, slat, t_lon, t_lat)
            else:
                lon_q, lat_q = t_lon, t_lat
                dims_q = ref[rlon].dims
                u_np, v_np = _manual_nearest_sample(src_t, su, sv, slon, slat, lon_q, lat_q)
            src_i = xr.Dataset({su: (dims_q, u_np), sv: (dims_q, v_np)})
    else:
        chunks: list[xr.Dataset] = []
        for tt in np.ravel(t_time):
            src_t = src.sel({stime: tt}, method="nearest")
            part = None
            if interp_method in {"auto", "linear"}:
                try:
                    part = src_t.interp(
                        {
                            slon: xr.DataArray(t_lon, dims=ref[rlon].dims),
                            slat: xr.DataArray(t_lat, dims=ref[rlat].dims),
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
                if t_lon.ndim == 1 and t_lat.ndim == 1:
                    dims_q = (ref[rlat].dims[0], ref[rlon].dims[0])
                    u_np, v_np = _manual_nearest_sample_rect(src_t, su, sv, slon, slat, t_lon, t_lat)
                else:
                    lon_q, lat_q = t_lon, t_lat
                    dims_q = ref[rlon].dims
                    u_np, v_np = _manual_nearest_sample(src_t, su, sv, slon, slat, lon_q, lat_q)
                part = xr.Dataset({su: (dims_q, u_np), sv: (dims_q, v_np)})
            part = part.expand_dims({stime: [tt]})
            chunks.append(part)
        src_i = xr.concat(chunks, dim=stime)

    try:
        u_src = src_i[su].to_numpy()
    except MemoryError:
        print("[WARN] streamed read for interpolated u due to OOM")
        u_src = _to_numpy_streamed(src_i[su], chunk_len=64)
    try:
        v_src = src_i[sv].to_numpy()
    except MemoryError:
        print("[WARN] streamed read for interpolated v due to OOM")
        v_src = _to_numpy_streamed(src_i[sv], chunk_len=64)
    try:
        u_ref = ref[ru].to_numpy()
    except MemoryError:
        print("[WARN] streamed read for HF u due to OOM")
        u_ref = _to_numpy_streamed(ref[ru], chunk_len=64)
    try:
        v_ref = ref[rv].to_numpy()
    except MemoryError:
        print("[WARN] streamed read for HF v due to OOM")
        v_ref = _to_numpy_streamed(ref[rv], chunk_len=64)

    u_src, _ = _scale_to_mps(u_src, str(src[su].attrs.get("units", "")))
    v_src, _ = _scale_to_mps(v_src, str(src[sv].attrs.get("units", "")))
    u_ref, _ = _scale_to_mps(u_ref, str(ref[ru].attrs.get("units", "")))
    v_ref, _ = _scale_to_mps(v_ref, str(ref[rv].attrs.get("units", "")))
    return u_src, v_src, u_ref, v_ref


def _accumulate(stats: RunningStats, u_pred: np.ndarray, v_pred: np.ndarray, u_true: np.ndarray, v_true: np.ndarray, max_abs_speed: float) -> None:
    mask = (
        np.isfinite(u_pred)
        & np.isfinite(v_pred)
        & np.isfinite(u_true)
        & np.isfinite(v_true)
        & (np.abs(u_pred) <= max_abs_speed)
        & (np.abs(v_pred) <= max_abs_speed)
        & (np.abs(u_true) <= max_abs_speed)
        & (np.abs(v_true) <= max_abs_speed)
    )
    up, vp, ut, vt = u_pred[mask], v_pred[mask], u_true[mask], v_true[mask]
    if up.size == 0:
        return
    sp = np.sqrt(up**2 + vp**2)
    st = np.sqrt(ut**2 + vt**2)

    stats.n += up.size
    stats.sum_du2 += float(np.sum((up - ut) ** 2))
    stats.sum_dv2 += float(np.sum((vp - vt) ** 2))
    stats.sum_dvec2 += float(np.sum((up - ut) ** 2 + (vp - vt) ** 2))
    stats.sum_dspeed2 += float(np.sum((sp - st) ** 2))
    stats.sum_up += float(np.sum(up))
    stats.sum_ut += float(np.sum(ut))
    stats.sum_up2 += float(np.sum(up**2))
    stats.sum_ut2 += float(np.sum(ut**2))
    stats.sum_uput += float(np.sum(up * ut))
    stats.sum_vp += float(np.sum(vp))
    stats.sum_vt += float(np.sum(vt))
    stats.sum_vp2 += float(np.sum(vp**2))
    stats.sum_vt2 += float(np.sum(vt**2))
    stats.sum_vpvt += float(np.sum(vp * vt))


def _finalize(stats: RunningStats) -> Metrics:
    if stats.n == 0:
        return Metrics(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    def corr(n: int, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float:
        num = n * sxy - sx * sy
        den_term = (n * sxx - sx * sx) * (n * syy - sy * sy)
        if den_term <= 1e-12:
            return np.nan
        den = np.sqrt(den_term)
        return float(num / den)

    return Metrics(
        rmse_u=float(np.sqrt(stats.sum_du2 / stats.n)),
        rmse_v=float(np.sqrt(stats.sum_dv2 / stats.n)),
        rmse_vec=float(np.sqrt(stats.sum_dvec2 / stats.n)),
        rmse_speed=float(np.sqrt(stats.sum_dspeed2 / stats.n)),
        corr_u=corr(stats.n, stats.sum_up, stats.sum_ut, stats.sum_up2, stats.sum_ut2, stats.sum_uput),
        corr_v=corr(stats.n, stats.sum_vp, stats.sum_vt, stats.sum_vp2, stats.sum_vt2, stats.sum_vpvt),
    )


def _print(name: str, m: Metrics) -> None:
    print(f"\n[{name}]")
    print(f"RMSE_u={m.rmse_u:.4f}, RMSE_v={m.rmse_v:.4f}, RMSE_vec={m.rmse_vec:.4f}, RMSE_speed={m.rmse_speed:.4f}")
    print(f"Corr_u={m.corr_u:.4f}, Corr_v={m.corr_v:.4f}")


def _small_scale_energy(u: np.ndarray, v: np.ndarray, max_abs_speed: float) -> float:
    arr_u = np.asarray(u)
    arr_v = np.asarray(v)
    if arr_u.ndim == 2:
        arr_u = arr_u[None, ...]
        arr_v = arr_v[None, ...]
    if arr_u.ndim < 3:
        return np.nan

    energies: list[float] = []
    for t in range(arr_u.shape[0]):
        uu = arr_u[t]
        vv = arr_v[t]
        mask = np.isfinite(uu) & np.isfinite(vv) & (np.abs(uu) <= max_abs_speed) & (np.abs(vv) <= max_abs_speed)
        if np.mean(mask) < 0.1:
            continue
        fill_u = float(np.nanmedian(uu[mask]))
        fill_v = float(np.nanmedian(vv[mask]))
        uu_f = np.where(mask, uu, fill_u)
        vv_f = np.where(mask, vv, fill_v)
        du_dy, du_dx = np.gradient(uu_f)
        dv_dy, dv_dx = np.gradient(vv_f)
        e_arr = du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2
        e = float(np.mean(e_arr[mask]))
        if np.isfinite(e):
            energies.append(e)
    return float(np.nanmedian(energies)) if energies else np.nan


def _safe_median(vals: list[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.median(arr))


def _save_snapshots(
    out_dir: str,
    file_tag: str,
    n_snap: int,
    t_vals: np.ndarray,
    uk: np.ndarray,
    vk: np.ndarray,
    um: np.ndarray,
    vm: np.ndarray,
    uc: np.ndarray,
    vc: np.ndarray,
    ut: np.ndarray,
    vt: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"Skip snapshot plot: matplotlib unavailable ({exc})")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if ut.ndim == 2:
        ut = ut[None, ...]
        vt = vt[None, ...]
        uk = uk[None, ...]
        vk = vk[None, ...]
        um = um[None, ...]
        vm = vm[None, ...]
        uc = uc[None, ...]
        vc = vc[None, ...]
        t_vals = np.array([t_vals[0] if np.size(t_vals) else 0], dtype=object)
    if ut.ndim < 3 or ut.shape[0] == 0:
        return

    idx = np.linspace(0, ut.shape[0] - 1, min(n_snap, ut.shape[0]), dtype=int)
    for k, ti in enumerate(idx):
        hf_spd = np.sqrt(ut[ti] ** 2 + vt[ti] ** 2)
        kan_spd = np.sqrt(uk[ti] ** 2 + vk[ti] ** 2)
        mlp_spd = np.sqrt(um[ti] ** 2 + vm[ti] ** 2)
        cm_spd = np.sqrt(uc[ti] ** 2 + vc[ti] ** 2)
        vmax = np.nanpercentile(hf_spd, 99)
        vmax = max(vmax, 1e-6)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, arr, title in [
            (axes[0, 0], hf_spd, "HF (truth)"),
            (axes[0, 1], kan_spd, "KAN recon"),
            (axes[1, 0], mlp_spd, "MLP recon"),
            (axes[1, 1], cm_spd, "CMEMS"),
        ]:
            im = ax.imshow(arr, origin="lower", cmap="turbo", vmin=0.0, vmax=vmax, aspect="auto")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        t_label = str(t_vals[ti]) if np.size(t_vals) else str(ti)
        fig.suptitle(f"Speed snapshot | {file_tag} | t={t_label}")
        fig.tight_layout()
        out_png = Path(out_dir) / f"{file_tag}_snap_{k+1:02d}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved snapshot: {out_png}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dir", default=r"D:\生产力\py程序\pythonProject\hf\202307_uswc_1km_rtv_sio")
    p.add_argument("--recon-kan", default=r"D:\data\recon_kan.nc")
    p.add_argument("--recon-mlp", default=r"D:\data\recon_mlp.nc")
    p.add_argument("--cmems", default=r"D:\data\uswc_2023_cmems_cleaned.nc")
    p.add_argument("--max-abs-speed", type=float, default=5.0)
    p.add_argument("--interp-method", choices=["auto", "linear", "nearest"], default="auto")
    p.add_argument("--snapshot-out-dir", default="hf_snapshots")
    p.add_argument("--snapshot-count", type=int, default=3)
    args = p.parse_args()

    hf_files = _list_hf_files(args.hf_dir)
    kan = _open_dataset_safe(args.recon_kan, "KAN reconstruction")
    mlp = _open_dataset_safe(args.recon_mlp, "MLP reconstruction")
    cmems = _open_dataset_safe(args.cmems, "CMEMS")

    s_kan, s_mlp, s_cmems = RunningStats(), RunningStats(), RunningStats()
    e_hf: list[float] = []
    e_kan: list[float] = []
    e_mlp: list[float] = []
    e_cmems: list[float] = []
    for i, hf_file in enumerate(hf_files, start=1):
        hf = _open_dataset_safe(hf_file, f"HF file {hf_file.name}")
        try:
            hf_use = hf
            last_mem_err: Exception | None = None
            for stride in (1, 2, 4, 8):
                try:
                    hf_use = _spatial_stride(hf, stride)
                    if stride > 1:
                        print(f"[WARN] OOM fallback: use HF spatial stride={stride} for {hf_file.name}")
                    uk, vk, ut, vt = _to_common_grid(kan, hf_use, interp_method=args.interp_method)
                    um, vm, _, _ = _to_common_grid(mlp, hf_use, interp_method=args.interp_method)
                    uc, vc, _, _ = _to_common_grid(cmems, hf_use, interp_method=args.interp_method)
                    break
                except MemoryError as exc:
                    last_mem_err = exc
                    if stride == 8:
                        raise
                    continue
            else:  # pragma: no cover
                if last_mem_err is not None:
                    raise last_mem_err
            t_name = _find(hf, TIME_CANDS, "time")
            t_vals = hf_use[t_name].values
            if i == 1:
                _save_snapshots(
                    out_dir=args.snapshot_out_dir,
                    file_tag=hf_file.stem,
                    n_snap=args.snapshot_count,
                    t_vals=t_vals,
                    uk=uk,
                    vk=vk,
                    um=um,
                    vm=vm,
                    uc=uc,
                    vc=vc,
                    ut=ut,
                    vt=vt,
                )
            e_hf.append(_small_scale_energy(ut, vt, args.max_abs_speed))
            e_kan.append(_small_scale_energy(uk, vk, args.max_abs_speed))
            e_mlp.append(_small_scale_energy(um, vm, args.max_abs_speed))
            e_cmems.append(_small_scale_energy(uc, vc, args.max_abs_speed))
            _accumulate(s_kan, uk, vk, ut, vt, args.max_abs_speed)
            _accumulate(s_mlp, um, vm, ut, vt, args.max_abs_speed)
            _accumulate(s_cmems, uc, vc, ut, vt, args.max_abs_speed)
            print(f"[HF] processed {i}/{len(hf_files)}: {hf_file.name}")
        finally:
            hf.close()

    m_kan = _finalize(s_kan)
    m_mlp = _finalize(s_mlp)
    m_cmems = _finalize(s_cmems)

    _print("KAN vs HF", m_kan)
    _print("MLP vs HF", m_mlp)
    _print("CMEMS vs HF", m_cmems)

    imp_kan = (m_cmems.rmse_vec - m_kan.rmse_vec) / (m_cmems.rmse_vec + 1e-12) * 100.0
    imp_mlp = (m_cmems.rmse_vec - m_mlp.rmse_vec) / (m_cmems.rmse_vec + 1e-12) * 100.0
    print(f"\nRMSE_vec improvement over CMEMS | KAN={imp_kan:.2f}% | MLP={imp_mlp:.2f}%")

    hf_e = _safe_median(e_hf)
    kan_e = _safe_median(e_kan)
    mlp_e = _safe_median(e_mlp)
    cmems_e = _safe_median(e_cmems)
    print("\n[Small-scale gradient energy] (closer to HF is better)")
    print(f"HF={hf_e:.4e} | KAN={kan_e:.4e} | MLP={mlp_e:.4e} | CMEMS={cmems_e:.4e}")
    kan.close()
    mlp.close()
    cmems.close()


if __name__ == "__main__":
    main()
