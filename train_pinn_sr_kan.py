# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset, Subset

DEFAULT_CMEMS_PATH = r"D:\data\uswc_2023_cmems_cleaned.nc"
DEFAULT_GDP_PATH = r"D:\download\uswc_drifter_6hour_2023_corrected.nc"

U_CANDS = ["u_bg", "u_clean", "u", "uo", "eastward_sea_water_velocity"]
V_CANDS = ["v_bg", "v_clean", "v", "vo", "northward_sea_water_velocity"]
LON_CANDS = ["lon", "longitude"]
LAT_CANDS = ["lat", "latitude"]
TIME_CANDS = ["time", "valid_time"]


def _find(ds: xr.Dataset, cands: list[str], label: str) -> str:
    for c in cands:
        if c in ds.variables or c in ds.coords:
            return c
    raise KeyError(f"Cannot find {label}; candidates={cands}, available={list(ds.variables)}")


def _load_gdp_df(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if ext == ".nc":
        return xr.open_dataset(path).to_dataframe().reset_index()
    raise ValueError(f"Unsupported GDP format: {path}")


def _inject_obs_from_gdp(u_obs: np.ndarray, v_obs: np.ndarray, time: np.ndarray, lat: np.ndarray, lon: np.ndarray, gdp_path: str):
    df = _load_gdp_df(gdp_path)
    col_map = {"lon": ["lon", "longitude"], "lat": ["lat", "latitude"], "time": ["time", "date", "datetime"], "u": ["u_obs", "u_corr", "u"], "v": ["v_obs", "v_corr", "v"]}
    picked = {}
    for k, cands in col_map.items():
        for c in cands:
            if c in df.columns:
                picked[k] = c
                break
    if len(picked) < 5:
        print("[WARN] GDP columns not complete; skip obs injection")
        return

    t = pd.to_datetime(df[picked["time"]], utc=True, errors="coerce").dt.tz_convert(None)
    valid = t.notna().to_numpy()
    t_vals = t[valid].to_numpy(dtype="datetime64[ns]")
    lon_vals = df[picked["lon"]].to_numpy(np.float64)[valid]
    lat_vals = df[picked["lat"]].to_numpy(np.float64)[valid]
    u_vals = df[picked["u"]].to_numpy(np.float32)[valid]
    v_vals = df[picked["v"]].to_numpy(np.float32)[valid]

    for tt, lo, la, uu, vv in zip(t_vals, lon_vals, lat_vals, u_vals, v_vals):
        if not np.isfinite([lo, la, uu, vv]).all():
            continue
        ti = int(np.argmin(np.abs(time - tt)))
        yi = int(np.argmin(np.abs(lat - la)))
        xi = int(np.argmin(np.abs(lon - lo)))
        u_obs[ti, yi, xi] = uu
        v_obs[ti, yi, xi] = vv


class SRDataset(Dataset):
    def __init__(self, data_path: str, gdp_path: str | None = None, max_obs_speed: float = 5.0):
        ds = xr.open_dataset(data_path)
        t_name = _find(ds, TIME_CANDS, "time")
        lat_name = _find(ds, LAT_CANDS, "lat")
        lon_name = _find(ds, LON_CANDS, "lon")

        self.time = pd.to_datetime(ds[t_name].values)
        lat = ds[lat_name].values
        lon = ds[lon_name].values

        if "u_bg" in ds and "v_bg" in ds and "u_obs" in ds and "v_obs" in ds:
            u_bg = ds["u_bg"].values.astype(np.float32)
            v_bg = ds["v_bg"].values.astype(np.float32)
            u_obs = ds["u_obs"].values.astype(np.float32)
            v_obs = ds["v_obs"].values.astype(np.float32)
        else:
            u_name = _find(ds, U_CANDS, "u")
            v_name = _find(ds, V_CANDS, "v")
            u_bg = ds[u_name].transpose(t_name, lat_name, lon_name).values.astype(np.float32)
            v_bg = ds[v_name].transpose(t_name, lat_name, lon_name).values.astype(np.float32)
            u_obs = np.full_like(u_bg, np.nan, dtype=np.float32)
            v_obs = np.full_like(v_bg, np.nan, dtype=np.float32)
            print(f"[INFO] using CMEMS background from {u_name}/{v_name}; obs initialized as NaN")
            if gdp_path and Path(gdp_path).exists():
                _inject_obs_from_gdp(u_obs, v_obs, self.time.to_numpy(dtype="datetime64[ns]"), lat, lon, gdp_path)

        bg_abs = np.abs(np.concatenate([u_bg[np.isfinite(u_bg)], v_bg[np.isfinite(v_bg)]]))
        p95_bg = float(np.nanpercentile(bg_abs, 95)) if bg_abs.size else np.nan
        if np.isfinite(p95_bg):
            if p95_bg > 2000.0:
                u_bg = u_bg / 1000.0
                v_bg = v_bg / 1000.0
                u_obs = u_obs / 1000.0
                v_obs = v_obs / 1000.0
                print(f"[WARN] background p95={p95_bg:.4f} is too large, auto scale bg/obs by /1000.")
            elif p95_bg > 20.0:
                u_bg = u_bg / 100.0
                v_bg = v_bg / 100.0
                u_obs = u_obs / 100.0
                v_obs = v_obs / 100.0
                print(f"[WARN] background p95={p95_bg:.4f} is too large, auto scale bg/obs by /100 (cm/s -> m/s).")

        bg_abs = np.abs(np.concatenate([u_bg[np.isfinite(u_bg)], v_bg[np.isfinite(v_bg)]]))
        obs_abs = np.abs(np.concatenate([u_obs[np.isfinite(u_obs)], v_obs[np.isfinite(v_obs)]]))
        p95_bg = float(np.nanpercentile(bg_abs, 95)) if bg_abs.size else np.nan
        p95_obs = float(np.nanpercentile(obs_abs, 95)) if obs_abs.size else np.nan
        if np.isfinite(p95_bg) and np.isfinite(p95_obs) and p95_bg > 1e-9:
            ratio = p95_obs / p95_bg
            if ratio > 50.0:
                u_obs = u_obs / 100.0
                v_obs = v_obs / 100.0
                print(f"[WARN] obs/bg p95 ratio={ratio:.2f}, auto scale obs by /100.")
                obs_abs = np.abs(np.concatenate([u_obs[np.isfinite(u_obs)], v_obs[np.isfinite(v_obs)]]))
                p95_obs = float(np.nanpercentile(obs_abs, 95)) if obs_abs.size else np.nan

        raw_mask = np.isfinite(u_obs) & np.isfinite(v_obs)
        speed_obs = np.sqrt(np.nan_to_num(u_obs, nan=0.0) ** 2 + np.nan_to_num(v_obs, nan=0.0) ** 2)
        phys_mask = speed_obs <= max_obs_speed
        self.obs_mask = raw_mask & phys_mask
        u_obs = np.where(self.obs_mask, u_obs, np.nan)
        v_obs = np.where(self.obs_mask, v_obs, np.nan)

        self.u_bg = np.nan_to_num(u_bg, nan=0.0, posinf=0.0, neginf=0.0)
        self.v_bg = np.nan_to_num(v_bg, nan=0.0, posinf=0.0, neginf=0.0)
        self.u_obs = np.nan_to_num(u_obs, nan=0.0, posinf=0.0, neginf=0.0)
        self.v_obs = np.nan_to_num(v_obs, nan=0.0, posinf=0.0, neginf=0.0)
        self.T = self.u_bg.shape[0]
        print(f"[CHECK] p95_bg={p95_bg:.4f}, p95_obs={p95_obs:.4f}, obs_points={int(self.obs_mask.sum())}")

    def __len__(self):
        return self.T

    def __getitem__(self, i):
        x = torch.from_numpy(np.stack([self.u_bg[i], self.v_bg[i]], axis=0))
        return x, torch.from_numpy(self.u_bg[i]), torch.from_numpy(self.v_bg[i]), torch.from_numpy(self.u_obs[i]), torch.from_numpy(self.v_obs[i]), torch.from_numpy(self.obs_mask[i])


class KANLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_basis: int = 8):
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


class KANPixelBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_basis):
        super().__init__()
        self.kan = KANLinear(in_ch, out_ch, num_basis)

    def forward(self, x):
        b, c, h, w = x.shape
        y = x.permute(0, 2, 3, 1).reshape(-1, c)
        y = self.kan(y)
        return y.reshape(b, h, w, -1).permute(0, 3, 1, 2)


class SmallSRKAN(nn.Module):
    def __init__(self, scale=2, width=48, num_basis=8):
        super().__init__()
        self.b1 = KANPixelBlock(2, width, num_basis)
        self.b2 = KANPixelBlock(width, width, num_basis)
        self.b3 = KANPixelBlock(width, width, num_basis)
        self.act = nn.Tanh()
        self.tail = nn.Conv2d(width, 2 * (scale ** 2), 3, padding=1)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.act(self.b1(x)); x = self.act(self.b2(x)); x = self.act(self.b3(x))
        return self.ps(self.tail(x))


def divergence_loss(u_hr, v_hr):
    du_dx = (u_hr[:, :, :, 2:] - u_hr[:, :, :, :-2]) * 0.5
    dv_dy = (v_hr[:, :, 2:, :] - v_hr[:, :, :-2, :]) * 0.5
    return ((du_dx[:, :, 1:-1, :] + dv_dy[:, :, :, 1:-1]) ** 2).mean()


def split_indices(times, train_ratio, seed, exclude_start=None, exclude_end=None):
    idx_all = np.arange(len(times))
    keep = np.ones(len(times), dtype=bool)
    if exclude_start and exclude_end:
        t0, t1 = pd.Timestamp(exclude_start), pd.Timestamp(exclude_end)
        leak = (times >= t0) & (times <= t1)
        keep = ~leak
        print(f"[INFO] excluded window: {exclude_start} ~ {exclude_end}, steps={int(leak.sum())}")
    idx = idx_all[keep]
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(len(idx) * train_ratio)
    tr, va = idx[:n_train], idx[n_train:]
    if len(tr) == 0 or len(va) == 0:
        raise ValueError("train/val split empty")
    return tr, va


def count_obs(ds, indices):
    return int(ds.obs_mask[indices].sum())


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    s_loss = s_bg = s_obs = s_phys = 0.0
    n_batch, n_obs = 0, 0
    for x, low_u, low_v, low_u_obs, low_v_obs, low_mask in loader:
        x = x.to(device).float(); low_u = low_u.to(device).float(); low_v = low_v.to(device).float()
        low_u_obs = low_u_obs.to(device).float(); low_v_obs = low_v_obs.to(device).float(); low_mask = low_mask.to(device).bool()
        y_hr = model(x); u_hr, v_hr = y_hr[:, 0:1], y_hr[:, 1:2]
        u_lr = F.avg_pool2d(u_hr, kernel_size=args.scale, stride=args.scale).squeeze(1)
        v_lr = F.avg_pool2d(v_hr, kernel_size=args.scale, stride=args.scale).squeeze(1)
        l_bg = ((u_lr - low_u) ** 2).mean() + ((v_lr - low_v) ** 2).mean()
        if low_mask.any():
            l_obs = args.w_obs_u * ((u_lr[low_mask] - low_u_obs[low_mask]) ** 2).mean() + args.w_obs_v * ((v_lr[low_mask] - low_v_obs[low_mask]) ** 2).mean()
            n_obs += int(low_mask.sum().item())
        else:
            l_obs = torch.tensor(0.0, device=device)
        l_phys = divergence_loss(u_hr, v_hr)
        loss = args.lambda_bg * l_bg + args.lambda_obs * l_obs + args.lambda_phys * l_phys
        s_loss += loss.item(); s_bg += l_bg.item(); s_obs += l_obs.item(); s_phys += l_phys.item(); n_batch += 1
    return {"loss": s_loss / max(n_batch, 1), "bg": s_bg / max(n_batch, 1), "obs": s_obs / max(n_batch, 1), "phys": s_phys / max(n_batch, 1), "obs_points": n_obs}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] device={device}")
    data_path = args.data if args.data else args.cmems
    ds = SRDataset(data_path, gdp_path=args.gdp, max_obs_speed=args.max_obs_speed)
    tr_idx, va_idx = split_indices(ds.time, args.train_ratio, args.seed, args.exclude_start, args.exclude_end)
    train_loader = DataLoader(Subset(ds, tr_idx.tolist()), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(ds, va_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[CHECK] train_obs_points={count_obs(ds, tr_idx)} val_obs_points={count_obs(ds, va_idx)}")

    model = SmallSRKAN(scale=args.scale, width=args.width, num_basis=args.num_basis).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    best_score, wait = np.inf, 0
    for ep in range(1, args.epochs + 1):
        model.train(); tr_loss_sum = 0.0; n_batch = 0
        for x, low_u, low_v, low_u_obs, low_v_obs, low_mask in train_loader:
            x = x.to(device).float(); low_u = low_u.to(device).float(); low_v = low_v.to(device).float()
            low_u_obs = low_u_obs.to(device).float(); low_v_obs = low_v_obs.to(device).float(); low_mask = low_mask.to(device).bool()
            y_hr = model(x); u_hr, v_hr = y_hr[:, 0:1], y_hr[:, 1:2]
            u_lr = F.avg_pool2d(u_hr, kernel_size=args.scale, stride=args.scale).squeeze(1)
            v_lr = F.avg_pool2d(v_hr, kernel_size=args.scale, stride=args.scale).squeeze(1)
            l_bg = ((u_lr - low_u) ** 2).mean() + ((v_lr - low_v) ** 2).mean()
            if low_mask.any():
                l_obs = args.w_obs_u * ((u_lr[low_mask] - low_u_obs[low_mask]) ** 2).mean() + args.w_obs_v * ((v_lr[low_mask] - low_v_obs[low_mask]) ** 2).mean()
            else:
                l_obs = torch.tensor(0.0, device=device)
            l_phys = divergence_loss(u_hr, v_hr)
            loss = args.lambda_bg * l_bg + args.lambda_obs * l_obs + args.lambda_phys * l_phys
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tr_loss_sum += loss.item(); n_batch += 1

        val = evaluate(model, val_loader, device, args)
        score = val["obs"] if np.isfinite(val["obs"]) else val["bg"]
        improved = ""
        if score < (best_score - args.min_delta):
            best_score, wait, improved = score, 0, "*"
            torch.save({"epoch": ep, "model": model.state_dict(), "best_score": best_score, "args": vars(args), "val_metrics": val}, out_dir / "best_sr.pt")
        else:
            wait += 1
        if ep % args.log_every == 0 or ep == 1:
            print(f"[Epoch {ep:04d}] train_loss={tr_loss_sum/max(n_batch,1):.6f} | val_bg={val['bg']:.6f} val_obs={val['obs']:.6f} val_phys={val['phys']:.6f} | score={score:.6f} best={best_score:.6f}{improved} | wait={wait}/{args.patience}")
        if wait >= args.patience:
            print(f"[EARLY STOP] no improvement for {args.patience} epochs.")
            break


def build_parser():
    p = argparse.ArgumentParser(description="Strict PINN-SR KAN training with weighted obs loss (u/v)")
    p.add_argument("--data", type=str, default="", help="combined training nc (u_bg/v_bg/u_obs/v_obs). empty -> use --cmems")
    p.add_argument("--cmems", type=str, default=DEFAULT_CMEMS_PATH, help="保持原来的默认路径")
    p.add_argument("--gdp", type=str, default=DEFAULT_GDP_PATH, help="保持原来的默认路径")
    p.add_argument("--out-dir", type=str, default=r"D:/data/output/checkpoints_sr_kan_strict_weighted")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--num-basis", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-obs", type=float, default=2.0)
    p.add_argument("--lambda-bg", type=float, default=0.3)
    p.add_argument("--lambda-phys", type=float, default=0.1)
    p.add_argument("--w-obs-u", type=float, default=1.0)
    p.add_argument("--w-obs-v", type=float, default=2.0)
    p.add_argument("--max-obs-speed", type=float, default=5.0, help="Mask obs points with speed above threshold (m/s)")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--exclude-start", type=str, default="2023-02-04 00:00:00")
    p.add_argument("--exclude-end", type=str, default="2023-03-05 23:59:59")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
