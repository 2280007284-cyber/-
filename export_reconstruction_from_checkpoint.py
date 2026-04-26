"""Export reconstructed flow NetCDF from trained MLP/KAN checkpoint.

Usage example:
python export_reconstruction_from_checkpoint.py \
  --model-type kan \
  --ckpt pinn_sr_kan_uv_best.pt \
  --cmems D:\\data\\uswc_2023_cmems_cleaned.nc \
  --out D:\\data\\recon_kan.nc
"""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

U_CANDS = ["u_clean", "u", "uo", "eastward_sea_water_velocity"]
V_CANDS = ["v_clean", "v", "vo", "northward_sea_water_velocity"]
LON_CANDS = ["lon", "longitude"]
LAT_CANDS = ["lat", "latitude"]
TIME_CANDS = ["time", "valid_time"]
U_SCALE_CANDS = ["u_clean", "u", "uo", "eastward_sea_water_velocity"]
V_SCALE_CANDS = ["v_clean", "v", "vo", "northward_sea_water_velocity"]


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _find(ds: xr.Dataset, cands: list[str], label: str) -> str:
    for c in cands:
        if c in ds.variables or c in ds.coords:
            return c
    raise KeyError(f"Cannot find {label}. candidates={cands}, available={list(ds.variables)}")


def _ensure_mps(uv: np.ndarray, units_hint: str = "") -> np.ndarray:
    if "cm/s" in units_hint or "cm s-1" in units_hint or "cm s^-1" in units_hint:
        return uv / 100.0
    if np.nanpercentile(np.abs(uv), 95) > 2.5:
        return uv / 100.0
    return uv


def _infer_norm_from_cmems(cmems_path: str, max_samples: int = 500_000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(cmems_path) as ds:
        lon_name = _find(ds, LON_CANDS, "lon")
        lat_name = _find(ds, LAT_CANDS, "lat")
        time_name = _find(ds, TIME_CANDS, "time")
        u_name = _find(ds, U_SCALE_CANDS, "u")
        v_name = _find(ds, V_SCALE_CANDS, "v")

        lon = ds[lon_name].values
        lat = ds[lat_name].values
        tim = ds[time_name].values.astype("datetime64[s]").astype(np.int64) / 3600.0
        u = ds[u_name].values.astype(np.float32)
        v = ds[v_name].values.astype(np.float32)
        nt, ny, nx = u.shape
        total = nt * ny * nx
        sample_n = min(max_samples, total)
        rng = np.random.default_rng(42)
        idx = rng.choice(total, size=sample_n, replace=False)

        ti = idx // (ny * nx)
        rem = idx % (ny * nx)
        yi = rem // nx
        xi = rem % nx
        xyt = np.column_stack([lon[xi], lat[yi], tim[ti]]).astype(np.float32)
        uv = np.column_stack([u[ti, yi, xi], v[ti, yi, xi]]).astype(np.float32)
        mask = np.isfinite(xyt).all(axis=1) & np.isfinite(uv).all(axis=1)
        xyt, uv = xyt[mask], uv[mask]
        units_hint = f"{str(ds[u_name].attrs.get('units','')).lower()}|{str(ds[v_name].attrs.get('units','')).lower()}"
        uv = _ensure_mps(uv, units_hint=units_hint)

    x_mu = xyt.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = (xyt.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    y_mu = uv.mean(axis=0, keepdims=True).astype(np.float32)
    y_std = (uv.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    return x_mu, x_std, y_mu, y_std


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int = 3, ff_dim: int = 32, scale: float = 6.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dim, ff_dim) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = 2 * torch.pi * (x @ self.B)
        return torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)


class MLP_PINN(nn.Module):
    def __init__(self, hidden: int = 128, depth: int = 6, ff_dim: int = 32):
        super().__init__()
        self.ff = FourierFeatures(3, ff_dim)
        layers: list[nn.Module] = [nn.Linear(ff_dim * 2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.ff(x))


class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_basis: int = 16):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(-2.0, 2.0, num_basis).repeat(in_dim, 1))
        self.log_sigma = nn.Parameter(torch.zeros(in_dim, 1))
        self.proj = nn.Linear(in_dim * num_basis, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_exp = x.unsqueeze(-1)
        c = self.centers.unsqueeze(0)
        sigma = torch.exp(self.log_sigma).unsqueeze(0) + 1e-6
        phi = torch.exp(-((x_exp - c) ** 2) / (2.0 * sigma**2))
        return self.proj(phi.reshape(x.shape[0], -1))


class KAN_PINN(nn.Module):
    def __init__(self, width: int = 128, depth: int = 4, num_basis: int = 16):
        super().__init__()
        layers: list[nn.Module] = [KANLayer(3, width, num_basis), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [KANLayer(width, width, num_basis), nn.Tanh()]
        layers += [KANLayer(width, 2, num_basis)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def export_one(
    model_type: str,
    ckpt_path: str,
    cmems_path: str,
    out_path: str,
    device: str,
    batch_size: int,
    max_norm_samples: int,
    log_every_batches: int,
    max_time_steps: int | None = None,
    trust_checkpoint: bool = True,
    upscale_factor: int = 1,
) -> None:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" in str(exc) and trust_checkpoint:
            _log(
                "[load] PyTorch>=2.6 detected weights_only restriction; "
                "retrying with weights_only=False because checkpoint is trusted."
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        else:
            raise
    # Backward compatibility: old checkpoints may be plain state_dict without stats.
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
        ckpt = {"model_state": state}

    if model_type == "mlp":
        model = MLP_PINN(hidden=int(ckpt.get("hidden", 128)), depth=int(ckpt.get("depth", 6)))
    else:
        model = KAN_PINN(
            width=int(ckpt.get("width", 128)),
            depth=int(ckpt.get("depth", 4)),
            num_basis=int(ckpt.get("num_basis", 16)),
        )

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    if all(k in ckpt for k in ["x_mu", "x_std", "y_mu", "y_std"]):
        x_mu = np.asarray(ckpt["x_mu"], dtype=np.float32)
        x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
        y_mu = np.asarray(ckpt["y_mu"], dtype=np.float32)
        y_std = np.asarray(ckpt["y_std"], dtype=np.float32)
    else:
        _log("Checkpoint missing normalization stats, inferring from CMEMS file for compatibility...")
        x_mu, x_std, y_mu, y_std = _infer_norm_from_cmems(cmems_path, max_samples=max_norm_samples)
    residual_mode = bool(ckpt.get("residual_mode", False))
    _log(f"[{model_type}] residual_mode={residual_mode}")

    with xr.open_dataset(cmems_path) as ds:
        lon_name = _find(ds, LON_CANDS, "lon")
        lat_name = _find(ds, LAT_CANDS, "lat")
        time_name = _find(ds, TIME_CANDS, "time")
        u_name = _find(ds, U_CANDS, "u")
        v_name = _find(ds, V_CANDS, "v")

        lon = ds[lon_name].values
        lat = ds[lat_name].values
        lon_out = lon
        lat_out = lat
        if upscale_factor > 1:
            lon_out = np.linspace(float(lon.min()), float(lon.max()), (len(lon) - 1) * upscale_factor + 1, dtype=np.float32)
            lat_out = np.linspace(float(lat.min()), float(lat.max()), (len(lat) - 1) * upscale_factor + 1, dtype=np.float32)
            _log(f"[{model_type}] using high-res grid: lon {len(lon)}->{len(lon_out)}, lat {len(lat)}->{len(lat_out)}")
        tim = ds[time_name].values.astype("datetime64[s]").astype(np.int64) / 3600.0
        time_coord = ds[time_name].values
        lat_coord = lat_out
        lon_coord = lon_out
        bg_u_all = ds[u_name].values.astype(np.float32)
        bg_v_all = ds[v_name].values.astype(np.float32)
        units_hint = f"{str(ds[u_name].attrs.get('units','')).lower()}|{str(ds[v_name].attrs.get('units','')).lower()}"
        bg_uv_all = _ensure_mps(np.stack([bg_u_all, bg_v_all], axis=-1), units_hint=units_hint)

        xx2d, yy2d = np.meshgrid(lon_out, lat_out, indexing="xy")

    if max_time_steps is not None:
        keep = max(1, min(int(max_time_steps), len(tim)))
        tim = tim[:keep]
        time_coord = time_coord[:keep]
        _log(f"[{model_type}] max-time-steps enabled: only exporting first {keep} steps")

    total_points = len(tim) * len(lat_out) * len(lon_out)
    _log(f"[{model_type}] workload: time={len(tim)}, lat={len(lat_out)}, lon={len(lon_out)}, total_points={total_points}")

    def predict_batched(x_arr: np.ndarray, device_name: str, bs: int) -> np.ndarray:
        preds = []
        total_batches = (x_arr.shape[0] + bs - 1) // bs
        t0 = time.time()
        with torch.no_grad():
            for bi, i in enumerate(range(0, x_arr.shape[0], bs), start=1):
                xb = torch.tensor(x_arr[i : i + bs], dtype=torch.float32, device=device_name)
                preds.append(model(xb).cpu().numpy())
                if log_every_batches > 0 and (bi % log_every_batches == 0 or bi == total_batches):
                    elapsed = time.time() - t0
                    _log(f"[{model_type}] batch {bi}/{total_batches}, elapsed={elapsed:.1f}s")
        return np.concatenate(preds, axis=0)

    u = np.empty((len(tim), len(lat_out), len(lon_out)), dtype=np.float32)
    v = np.empty((len(tim), len(lat_out), len(lon_out)), dtype=np.float32)
    for ti, t_hour in enumerate(tim):
        xyt_raw = np.column_stack(
            [xx2d.ravel(), yy2d.ravel(), np.full(xx2d.size, t_hour, dtype=np.float32)]
        ).astype(np.float32)
        xyt = (xyt_raw - x_mu) / x_std
        try:
            pred = predict_batched(xyt, device, batch_size)
        except torch.OutOfMemoryError:
            if device.startswith("cuda"):
                _log("CUDA OOM during export, falling back to CPU batched inference...")
                model.to("cpu")
                pred = predict_batched(xyt, "cpu", max(5000, batch_size // 5))
            else:
                raise
        if residual_mode:
            if upscale_factor > 1:
                bg_u_t = xr.DataArray(
                    bg_uv_all[ti, :, :, 0],
                    coords={lat_name: lat, lon_name: lon},
                    dims=(lat_name, lon_name),
                ).interp({lat_name: lat_out, lon_name: lon_out}, method="linear").values
                bg_v_t = xr.DataArray(
                    bg_uv_all[ti, :, :, 1],
                    coords={lat_name: lat, lon_name: lon},
                    dims=(lat_name, lon_name),
                ).interp({lat_name: lat_out, lon_name: lon_out}, method="linear").values
                bg_uv_t = np.column_stack([bg_u_t.ravel(), bg_v_t.ravel()])
            else:
                bg_uv_t = bg_uv_all[ti].reshape(-1, 2)
            uv = pred * y_std + bg_uv_t
        else:
            uv = pred * y_std + y_mu
        u[ti] = uv[:, 0].reshape(len(lat_out), len(lon_out))
        v[ti] = uv[:, 1].reshape(len(lat_out), len(lon_out))
        _log(f"[{model_type}] time-step {ti + 1}/{len(tim)} done")

    out = xr.Dataset(
        {
            "u_recon": ((time_name, lat_name, lon_name), u.astype(np.float32)),
            "v_recon": ((time_name, lat_name, lon_name), v.astype(np.float32)),
        },
        coords={time_name: time_coord, lat_name: lat_coord, lon_name: lon_coord},
        attrs={"model_type": model_type, "checkpoint": ckpt_path},
    )
    out.to_netcdf(out_path)
    _log(f"Saved reconstructed flow to: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["mlp", "kan", "both"], default="both")
    p.add_argument("--ckpt", default=None, help="Single-model mode only: checkpoint path")
    p.add_argument("--ckpt-mlp", default="pinn_sr_mlp_uv_best.pt")
    p.add_argument("--ckpt-kan", default="pinn_sr_kan_uv_best.pt")
    p.add_argument("--cmems", default=r"D:\data\uswc_2023_cmems_cleaned.nc")
    p.add_argument("--out", default=None, help="Single-model mode only: output NetCDF path")
    p.add_argument("--out-mlp", default=r"D:\data\recon_mlp.nc")
    p.add_argument("--out-kan", default=r"D:\data\recon_kan.nc")
    # Backward/typo compatibility: some users used --recon as output path.
    p.add_argument("--recon", default=None, help="Alias of --out (compatibility)")
    # Ignored compatibility args (from compare script).
    p.add_argument("--gdp", default=None, help=argparse.SUPPRESS)
    p.add_argument("--val-days", default=None, help=argparse.SUPPRESS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=50000, help="Inference batch size to avoid OOM")
    p.add_argument("--max-norm-samples", type=int, default=500000, help="Max sampled points for norm fallback")
    p.add_argument("--log-every-batches", type=int, default=20, help="Progress print interval per inference batch")
    p.add_argument("--max-time-steps", type=int, default=None, help="Only export first N time steps (debug/smoke)")
    p.add_argument("--upscale-factor", type=int, default=1, help=">1 to output higher-resolution lon/lat grid")
    p.add_argument(
        "--untrusted-checkpoint",
        action="store_true",
        help="Disable unsafe fallback load (weights_only=False) for PyTorch>=2.6",
    )
    args = p.parse_args()
    _log(
        f"Start export: model_type={args.model_type}, device={args.device}, "
        f"batch_size={args.batch_size}, cmems={args.cmems}"
    )

    if args.model_type == "both":
        if args.ckpt is not None or args.out is not None or args.recon is not None:
            raise ValueError(
                "--model-type both 时不能使用 --ckpt/--out/--recon；"
                "请改用 --ckpt-kan/--ckpt-mlp 和 --out-kan/--out-mlp"
            )
        export_one(
            "kan",
            args.ckpt_kan,
            args.cmems,
            args.out_kan,
            args.device,
            args.batch_size,
            args.max_norm_samples,
            args.log_every_batches,
            args.max_time_steps,
            not args.untrusted_checkpoint,
            args.upscale_factor,
        )
        export_one(
            "mlp",
            args.ckpt_mlp,
            args.cmems,
            args.out_mlp,
            args.device,
            args.batch_size,
            args.max_norm_samples,
            args.log_every_batches,
            args.max_time_steps,
            not args.untrusted_checkpoint,
            args.upscale_factor,
        )
    else:
        default_ckpt = "pinn_sr_kan_uv_best.pt" if args.model_type == "kan" else "pinn_sr_mlp_uv_best.pt"
        default_out = r"D:\data\recon_kan.nc" if args.model_type == "kan" else r"D:\data\recon_mlp.nc"
        ckpt_path = args.ckpt or default_ckpt
        out_path = args.out or args.recon or default_out
        export_one(
            args.model_type,
            ckpt_path,
            args.cmems,
            out_path,
            args.device,
            args.batch_size,
            args.max_norm_samples,
            args.log_every_batches,
            args.max_time_steps,
            not args.untrusted_checkpoint,
            args.upscale_factor,
        )


if __name__ == "__main__":
    main()
