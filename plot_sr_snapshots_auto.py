from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr


class SmallSRNet(nn.Module):
    """需与训练脚本结构一致。"""

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


def pick_top_times(times: pd.DatetimeIndex, obs_count: np.ndarray, top_n: int, min_gap_hours: int) -> list[int]:
    order = np.argsort(-obs_count)
    picked: list[int] = []
    picked_t: list[pd.Timestamp] = []
    for i in order:
        if obs_count[i] <= 0:
            continue
        t = times[i]
        if all(abs((t - pt).total_seconds()) >= min_gap_hours * 3600 for pt in picked_t):
            picked.append(int(i))
            picked_t.append(t)
        if len(picked) >= top_n:
            break
    return picked


def upsample_bilinear(u_lr: np.ndarray, v_lr: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([u_lr, v_lr], axis=0)[None, ...].astype(np.float32)
    with torch.no_grad():
        y = F.interpolate(torch.from_numpy(x), scale_factor=scale, mode="bilinear", align_corners=False).numpy()[0]
    return y[0], y[1]


def draw_quiver(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, u: np.ndarray, v: np.ndarray, step: int = 8) -> None:
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


def _pick_name(ds: xr.Dataset, names: list[str]) -> str:
    for n in names:
        if n in ds.variables or n in ds.coords:
            return n
    raise KeyError(f"None of candidates found: {names}")


def main(args: argparse.Namespace) -> None:
    set_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.data)
    t_name = _pick_name(ds, ["time", "valid_time"])
    lon_name = _pick_name(ds, ["longitude", "lon"])
    lat_name = _pick_name(ds, ["latitude", "lat"])
    u_bg_name = _pick_name(ds, ["u_bg", "u_clean", "u"])
    v_bg_name = _pick_name(ds, ["v_bg", "v_clean", "v"])
    u_obs_name = _pick_name(ds, ["u_obs", "u_corr", "u"])
    v_obs_name = _pick_name(ds, ["v_obs", "v_corr", "v"])

    times = pd.to_datetime(ds[t_name].values)
    lon_lr = ds[lon_name].values
    lat_lr = ds[lat_name].values

    u_bg = np.nan_to_num(ds[u_bg_name].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    v_bg = np.nan_to_num(ds[v_bg_name].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    u_obs = ds[u_obs_name].values.astype(np.float32)
    v_obs = ds[v_obs_name].values.astype(np.float32)

    if args.start and args.end:
        t0, t1 = pd.Timestamp(args.start), pd.Timestamp(args.end)
        keep = (times >= t0) & (times <= t1)
        if keep.sum() == 0:
            raise ValueError("指定时间窗内没有样本")
        times = times[keep]
        u_bg, v_bg, u_obs, v_obs = u_bg[keep], v_bg[keep], u_obs[keep], v_obs[keep]

    obs_mask = np.isfinite(u_obs) & np.isfinite(v_obs)
    obs_count = obs_mask.reshape(obs_mask.shape[0], -1).sum(axis=1)
    idx = pick_top_times(times, obs_count, args.top_n, args.min_gap_hours)
    if not idx:
        raise ValueError("没有可视化时刻（观测点全为0）")

    sel_df = pd.DataFrame({"time": times[idx], "obs_points": obs_count[idx]})
    sel_df.to_csv(out_dir / "selected_times.csv", index=False, encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = SmallSRNet(scale=args.scale, ch=args.channels).to(device)
    state = ckpt.get("model", ckpt.get("model_state_dict"))
    if state is None:
        raise KeyError("checkpoint 中不存在 model/model_state_dict")
    model.load_state_dict(state, strict=True)
    model.eval()

    lon_hr = np.linspace(lon_lr.min(), lon_lr.max(), len(lon_lr) * args.scale)
    lat_hr = np.linspace(lat_lr.min(), lat_lr.max(), len(lat_lr) * args.scale)

    for i in idx:
        u_lr = u_bg[i]
        v_lr = v_bg[i]

        u_bi, v_bi = upsample_bilinear(u_lr, v_lr, args.scale)

        x = np.stack([u_lr, v_lr], axis=0)[None, ...].astype(np.float32)
        with torch.no_grad():
            y_hr = model(torch.from_numpy(x).to(device))
            if torch.isnan(y_hr).any() or torch.isinf(y_hr).any():
                raise ValueError(f"模型输出异常：time={times[i]}")
        u_sr = y_hr[0, 0].cpu().numpy()
        v_sr = y_hr[0, 1].cpu().numpy()

        # 同网格展示：CMEMS 用双线性上采样仅用于显示对齐
        u_cm = u_bi.copy()
        v_cm = v_bi.copy()

        sp_cm = np.sqrt(u_cm**2 + v_cm**2)
        sp_bi = np.sqrt(u_bi**2 + v_bi**2)
        sp_sr = np.sqrt(u_sr**2 + v_sr**2)
        sp_diff = sp_sr - sp_bi

        vmax = np.nanpercentile(np.concatenate([sp_cm.ravel(), sp_bi.ravel(), sp_sr.ravel()]), 99)
        dmax = np.nanpercentile(np.abs(sp_diff), 99)

        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), constrained_layout=True)

        im1 = axes[0, 0].pcolormesh(lon_hr, lat_hr, sp_cm, shading="auto", cmap="turbo", vmin=0, vmax=vmax)
        draw_quiver(axes[0, 0], lon_hr, lat_hr, u_cm, v_cm, args.quiver_step)
        axes[0, 0].set_title("CMEMS (displayed on HR grid)")

        im2 = axes[0, 1].pcolormesh(lon_hr, lat_hr, sp_bi, shading="auto", cmap="turbo", vmin=0, vmax=vmax)
        draw_quiver(axes[0, 1], lon_hr, lat_hr, u_bi, v_bi, args.quiver_step)
        axes[0, 1].set_title("Bilinear upsampling")

        im3 = axes[1, 0].pcolormesh(lon_hr, lat_hr, sp_sr, shading="auto", cmap="turbo", vmin=0, vmax=vmax)
        draw_quiver(axes[1, 0], lon_hr, lat_hr, u_sr, v_sr, args.quiver_step)
        axes[1, 0].set_title("PINN-SR (HR)")

        im4 = axes[1, 1].pcolormesh(lon_hr, lat_hr, sp_diff, shading="auto", cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        axes[1, 1].set_title("Speed difference: PINN-SR - Bilinear")

        for ax in axes.ravel():
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        cb1 = fig.colorbar(im3, ax=[axes[0, 0], axes[0, 1], axes[1, 0]], shrink=0.86, pad=0.02)
        cb1.set_label("Speed (m/s)")
        cb2 = fig.colorbar(im4, ax=axes[1, 1], shrink=0.86, pad=0.02)
        cb2.set_label("ΔSpeed (m/s)")

        fig.suptitle(f"Snapshot {pd.Timestamp(times[i]).strftime('%Y-%m-%d %H:%M:%S')} | obs={int(obs_count[i])}")

        stem = out_dir / f"snapshot_{pd.Timestamp(times[i]).strftime('%Y%m%d_%H%M%S')}"
        fig.savefig(stem.with_suffix(".png"))
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)
        print(f"[INFO] saved: {stem.with_suffix('.png')}")

    print("[INFO] done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动选时刻并输出 PINN-SR 流场快照")
    parser.add_argument("--data", type=str, default=r"D:/data/output/train_data_2023.nc")
    parser.add_argument("--ckpt", type=str, default=r"D:/data/output/checkpoints_sr_strict_weighted/best_sr.pt")
    parser.add_argument("--out-dir", type=str, default=r"D:/data/output/snapshots_sr_auto")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--top-n", type=int, default=6, help="自动选择时刻数量")
    parser.add_argument("--min-gap-hours", type=int, default=24, help="选中时刻最小间隔（小时）")
    parser.add_argument("--start", type=str, default=None, help="可选：起始时间")
    parser.add_argument("--end", type=str, default=None, help="可选：结束时间")
    parser.add_argument("--quiver-step", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
