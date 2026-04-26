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

COLORS = {
    "cmems": "#0072B2",
    "pinn": "#D55E00",
    "neutral": "#4D4D4D",
    "bg": "#F5F5F5",
}


def set_pub_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.facecolor": COLORS["bg"],
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.25,
            "axes.linewidth": 0.8,
        }
    )


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


def _adapt_state_dict_keys_for_srnet(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compatibility shim for checkpoints saved with older/newer module names.

    Some checkpoints use `body.*` while current eval model expects `head.*`.
    """
    if any(k.startswith("body.") for k in state.keys()) and not any(k.startswith("head.") for k in state.keys()):
        return {("head." + k[len("body."):]) if k.startswith("body.") else k: v for k, v in state.items()}
    if any(k.startswith("head.") for k in state.keys()) and not any(k.startswith("body.") for k in state.keys()):
        return state
    return state


def mask2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = mask2(a, b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() else np.nan


def mae(a: np.ndarray, b: np.ndarray) -> float:
    m = mask2(a, b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.sum() else np.nan


def bias(a: np.ndarray, b: np.ndarray) -> float:
    m = mask2(a, b)
    return float(np.mean(a[m] - b[m])) if m.sum() else np.nan


def corr(a: np.ndarray, b: np.ndarray) -> float:
    m = mask2(a, b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan


def r2(a: np.ndarray, b: np.ndarray) -> float:
    m = mask2(a, b)
    if m.sum() < 3:
        return np.nan
    pred, obs = a[m], b[m]
    ss_res = np.sum((pred - obs) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan


def willmott_d(a: np.ndarray, b: np.ndarray) -> float:
    m = mask2(a, b)
    if m.sum() < 3:
        return np.nan
    pred, obs = a[m], b[m]
    den = np.sum((np.abs(pred - np.mean(obs)) + np.abs(obs - np.mean(obs))) ** 2)
    num = np.sum((pred - obs) ** 2)
    return float(1 - num / den) if den > 1e-12 else np.nan


def direction_error_deg(u_m: np.ndarray, v_m: np.ndarray, u_o: np.ndarray, v_o: np.ndarray) -> float:
    m = np.isfinite(u_m) & np.isfinite(v_m) & np.isfinite(u_o) & np.isfinite(v_o)
    if m.sum() == 0:
        return np.nan
    ang_m = np.degrees(np.arctan2(v_m[m], u_m[m]))
    ang_o = np.degrees(np.arctan2(v_o[m], u_o[m]))
    d = (ang_m - ang_o + 180) % 360 - 180
    return float(np.mean(np.abs(d)))


def scatter_index(model_speed: np.ndarray, obs_speed: np.ndarray) -> float:
    return rmse(model_speed, obs_speed) / (np.nanmean(obs_speed) + 1e-12)


def improve(base: float, new: float) -> float:
    if (not np.isfinite(base)) or base <= 1e-12 or (not np.isfinite(new)):
        return np.nan
    return (base - new) / base * 100.0


def grad_mag(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    du_dy, du_dx = np.gradient(u, axis=(1, 2))
    dv_dy, dv_dx = np.gradient(v, axis=(1, 2))
    return np.sqrt(du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2)


def vorticity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    du_dy = np.gradient(u, axis=1)
    dv_dx = np.gradient(v, axis=2)
    return dv_dx - du_dy


def divergence(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    du_dx = np.gradient(u, axis=2)
    dv_dy = np.gradient(v, axis=1)
    return du_dx + dv_dy


def hann2d(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w))


def isotropic_ke_spectrum(u2d: np.ndarray, v2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u2d, dtype=np.float64)
    v = np.asarray(v2d, dtype=np.float64)
    h, w = u.shape
    u = (u - np.nanmean(u)) * hann2d(h, w)
    v = (v - np.nanmean(v)) * hann2d(h, w)
    uh = np.fft.fft2(u)
    vh = np.fft.fft2(v)
    e2d = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2) / (h * w)

    kx = np.fft.fftfreq(w) * w
    ky = np.fft.fftfreq(h) * h
    kxg, kyg = np.meshgrid(kx, ky)
    kr = np.sqrt(kxg**2 + kyg**2)

    kmax = int(np.floor(kr.max()))
    bins = np.arange(0, kmax + 1)
    ek = np.full_like(bins, np.nan, dtype=np.float64)
    kr_i = np.rint(kr).astype(int)

    for k in bins:
        m = kr_i == k
        if np.any(m):
            ek[k] = np.nanmean(e2d[m])

    k = bins[1:].astype(float)
    e = ek[1:]
    valid = np.isfinite(e) & (e > 0)
    return k[valid], e[valid]


def mean_spectrum(u3d: np.ndarray, v3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_ref = None
    rows = []
    for t in range(u3d.shape[0]):
        k, e = isotropic_ke_spectrum(u3d[t], v3d[t])
        if k_ref is None:
            k_ref = k
            rows.append(e)
        else:
            rows.append(np.interp(k_ref, k, e, left=np.nan, right=np.nan))
    e_mean = np.nanmean(np.array(rows), axis=0)
    valid = np.isfinite(e_mean) & (e_mean > 0)
    return k_ref[valid], e_mean[valid]


def slope_loglog(k: np.ndarray, e: np.ndarray, frac_lo: float = 0.2, frac_hi: float = 0.6) -> float:
    n = len(k)
    i1 = max(int(n * frac_lo), 1)
    i2 = max(int(n * frac_hi), i1 + 2)
    kk = k[i1:i2]
    ee = e[i1:i2]
    valid = (kk > 0) & (ee > 0) & np.isfinite(ee)
    if valid.sum() < 3:
        return np.nan
    a, _ = np.polyfit(np.log(kk[valid]), np.log(ee[valid]), 1)
    return float(a)


def integrated_ratio(k: np.ndarray, e_sr: np.ndarray, e_bi: np.ndarray, lo: float, hi: float, eps: float = 1e-10) -> float:
    n = len(k)
    i1 = max(int(n * lo), 1)
    i2 = max(int(n * hi), i1 + 1)
    a = e_sr[i1:i2]
    b = e_bi[i1:i2]
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > eps)
    if valid.sum() == 0:
        return np.nan
    return float(np.sum(a[valid]) / np.sum(b[valid]))


def save_fig(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_grouped_bars(df_long: pd.DataFrame, title: str, ylabel: str, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(df_long))
    w = 0.35
    b1 = ax.bar(x - w / 2, df_long["CMEMS"], w, color=COLORS["cmems"], label="CMEMS")
    b2 = ax.bar(x + w / 2, df_long["PINN-SR"], w, color=COLORS["pinn"], label="PINN-SR")
    ax.set_xticks(x)
    ax.set_xticklabels(df_long["metric"], rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    save_fig(fig, out_base)


def plot_hist_pair(a: np.ndarray, b: np.ndarray, name: str, xlabel: str, out_base: Path) -> None:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    q1 = np.nanpercentile(np.concatenate([a, b]), 1)
    q99 = np.nanpercentile(np.concatenate([a, b]), 99)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.hist(a, bins=80, range=(q1, q99), density=True, alpha=0.5, color=COLORS["cmems"], label="CMEMS")
    ax.hist(b, bins=80, range=(q1, q99), density=True, alpha=0.5, color=COLORS["pinn"], label="PINN-SR")
    ax.set_title(name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    save_fig(fig, out_base)


def main(args: argparse.Namespace) -> None:
    set_pub_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.data)
    times = pd.to_datetime(ds["time"].values)
    u_bg = np.nan_to_num(ds["u_bg"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    v_bg = np.nan_to_num(ds["v_bg"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    u_obs = ds["u_obs"].values.astype(np.float32)
    v_obs = ds["v_obs"].values.astype(np.float32)

    if args.start and args.end:
        t0, t1 = pd.Timestamp(args.start), pd.Timestamp(args.end)
        keep = (times >= t0) & (times <= t1)
        if keep.sum() == 0:
            raise ValueError("No samples in selected window")
        u_bg, v_bg, u_obs, v_obs = u_bg[keep], v_bg[keep], u_obs[keep], v_obs[keep]

    t = u_bg.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = SmallSRNet(scale=args.scale, ch=args.channels).to(device)
    state = ckpt.get("model", ckpt.get("model_state_dict"))
    if state is None:
        raise KeyError("Checkpoint missing model/model_state_dict")
    state = _adapt_state_dict_keys_for_srnet(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    u_rec = np.zeros_like(u_bg, dtype=np.float32)
    v_rec = np.zeros_like(v_bg, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, t, args.batch_size):
            j = min(i + args.batch_size, t)
            x_t = torch.from_numpy(np.stack([u_bg[i:j], v_bg[i:j]], axis=1)).to(device)
            y_hr = model(x_t)
            u_hr, v_hr = y_hr[:, 0:1], y_hr[:, 1:2]
            u_rec[i:j] = F.avg_pool2d(u_hr, kernel_size=args.scale, stride=args.scale).squeeze(1).cpu().numpy()
            v_rec[i:j] = F.avg_pool2d(v_hr, kernel_size=args.scale, stride=args.scale).squeeze(1).cpu().numpy()

    metric_names = [
        "MAE_u", "MAE_v", "MAE_speed",
        "RMSE_u", "RMSE_v", "RMSE_speed",
        "Bias_u", "Bias_v", "Bias_speed",
        "SI_u", "SI_v", "SI_speed",
        "R_u", "R_v", "R_speed",
        "R2_u", "R2_v", "R2_speed",
        "Willmott_u", "Willmott_v", "Willmott_speed",
        "DirErr_deg",
    ]
    m = np.isfinite(u_obs) & np.isfinite(v_obs) & np.isfinite(u_rec) & np.isfinite(v_rec)
    if int(m.sum()) == 0:
        msg = (
            "No valid observation points in selected period. "
            "请检查 --start/--end 是否覆盖观测，或重建 data.nc 时增大 "
            "--max-time-diff-hours（建议 6~12），并确认 u_obs/v_obs 非空。"
        )
        if args.allow_no_obs:
            print(f"[WARN] {msg} obs-based metrics will be NaN because --allow-no-obs is enabled.")
            cmems = {k: np.nan for k in metric_names}
            pinn = {k: np.nan for k in metric_names}
        else:
            raise ValueError(msg)
    else:
        ub, vb = u_bg[m], v_bg[m]
        um, vm = u_rec[m], v_rec[m]
        uo, vo = u_obs[m], v_obs[m]

        sp_bg = np.sqrt(ub**2 + vb**2)
        sp_sr = np.sqrt(um**2 + vm**2)
        sp_obs = np.sqrt(uo**2 + vo**2)

        cmems = {
            "MAE_u": mae(ub, uo), "MAE_v": mae(vb, vo), "MAE_speed": mae(sp_bg, sp_obs),
            "RMSE_u": rmse(ub, uo), "RMSE_v": rmse(vb, vo), "RMSE_speed": rmse(sp_bg, sp_obs),
            "Bias_u": bias(ub, uo), "Bias_v": bias(vb, vo), "Bias_speed": bias(sp_bg, sp_obs),
            "SI_u": scatter_index(np.abs(ub), np.abs(uo)), "SI_v": scatter_index(np.abs(vb), np.abs(vo)), "SI_speed": scatter_index(sp_bg, sp_obs),
            "R_u": corr(ub, uo), "R_v": corr(vb, vo), "R_speed": corr(sp_bg, sp_obs),
            "R2_u": r2(ub, uo), "R2_v": r2(vb, vo), "R2_speed": r2(sp_bg, sp_obs),
            "Willmott_u": willmott_d(ub, uo), "Willmott_v": willmott_d(vb, vo), "Willmott_speed": willmott_d(sp_bg, sp_obs),
            "DirErr_deg": direction_error_deg(ub, vb, uo, vo),
        }
        pinn = {
            "MAE_u": mae(um, uo), "MAE_v": mae(vm, vo), "MAE_speed": mae(sp_sr, sp_obs),
            "RMSE_u": rmse(um, uo), "RMSE_v": rmse(vm, vo), "RMSE_speed": rmse(sp_sr, sp_obs),
            "Bias_u": bias(um, uo), "Bias_v": bias(vm, vo), "Bias_speed": bias(sp_sr, sp_obs),
            "SI_u": scatter_index(np.abs(um), np.abs(uo)), "SI_v": scatter_index(np.abs(vm), np.abs(vo)), "SI_speed": scatter_index(sp_sr, sp_obs),
            "R_u": corr(um, uo), "R_v": corr(vm, vo), "R_speed": corr(sp_sr, sp_obs),
            "R2_u": r2(um, uo), "R2_v": r2(vm, vo), "R2_speed": r2(sp_sr, sp_obs),
            "Willmott_u": willmott_d(um, uo), "Willmott_v": willmott_d(vm, vo), "Willmott_speed": willmott_d(sp_sr, sp_obs),
            "DirErr_deg": direction_error_deg(um, vm, uo, vo),
        }

    metrics_df = pd.DataFrame({
        "metric": list(cmems.keys()),
        "CMEMS": [cmems[k] for k in cmems],
        "PINN-SR": [pinn[k] for k in cmems],
        "improve_%(for_lower_better_only)": [
            improve(cmems[k], pinn[k]) if any(key in k for key in ["MAE", "RMSE", "Bias", "SI", "DirErr"]) else np.nan
            for k in cmems
        ],
    })
    metrics_df.to_csv(out_dir / "metrics_all.csv", index=False, encoding="utf-8")

    err_metrics = ["MAE_u", "MAE_v", "MAE_speed", "RMSE_u", "RMSE_v", "RMSE_speed", "Bias_u", "Bias_v", "Bias_speed", "SI_u", "SI_v", "SI_speed"]
    plot_grouped_bars(metrics_df[metrics_df["metric"].isin(err_metrics)], "Error Metrics", "Value", out_dir / "fig_error_metrics")
    cons_metrics = ["R_u", "R_v", "R_speed", "R2_u", "R2_v", "R2_speed", "Willmott_u", "Willmott_v", "Willmott_speed", "DirErr_deg"]
    plot_grouped_bars(metrics_df[metrics_df["metric"].isin(cons_metrics)], "Consistency / Correlation Metrics", "Value", out_dir / "fig_consistency_metrics")

    g_bg = grad_mag(u_bg, v_bg); g_sr = grad_mag(u_rec, v_rec)
    z_bg = vorticity(u_bg, v_bg); z_sr = vorticity(u_rec, v_rec)
    d_bg = divergence(u_bg, v_bg); d_sr = divergence(u_rec, v_rec)

    struct_df = pd.DataFrame({
        "metric": ["MAE_|∇U|", "RMSE_|∇U|", "MAE_vorticity", "RMSE_vorticity", "MAE_divergence", "RMSE_divergence"],
        "value": [mae(g_sr, g_bg), rmse(g_sr, g_bg), mae(z_sr, z_bg), rmse(z_sr, z_bg), mae(d_sr, d_bg), rmse(d_sr, d_bg)],
    })
    struct_df.to_csv(out_dir / "metrics_structural.csv", index=False, encoding="utf-8")

    plot_hist_pair(g_bg.ravel(), g_sr.ravel(), "Gradient Magnitude Distribution", r"|∇U|", out_dir / "fig_hist_gradmag")
    plot_hist_pair(z_bg.ravel(), z_sr.ravel(), "Vorticity Distribution", r"ζ", out_dir / "fig_hist_vorticity")
    plot_hist_pair(d_bg.ravel(), d_sr.ravel(), "Divergence Distribution", r"div", out_dir / "fig_hist_divergence")

    nt = u_bg.shape[0]
    if args.spectrum_times > 0 and args.spectrum_times < nt:
        rng = np.random.default_rng(args.seed)
        tidx = np.sort(rng.choice(np.arange(nt), size=args.spectrum_times, replace=False))
    else:
        tidx = np.arange(nt)

    u_sr_hr, v_sr_hr, u_bi_hr, v_bi_hr = [], [], [], []
    with torch.no_grad():
        for i in tidx:
            x_t = torch.from_numpy(np.stack([u_bg[i], v_bg[i]], axis=0)[None, ...]).to(device)
            y_hr = model(x_t)
            y_bi = F.interpolate(x_t, scale_factor=args.scale, mode="bilinear", align_corners=False)
            u_sr_hr.append(y_hr[0, 0].cpu().numpy()); v_sr_hr.append(y_hr[0, 1].cpu().numpy())
            u_bi_hr.append(y_bi[0, 0].cpu().numpy()); v_bi_hr.append(y_bi[0, 1].cpu().numpy())

    k_sr, e_sr = mean_spectrum(np.asarray(u_sr_hr), np.asarray(v_sr_hr))
    k_bi, e_bi = mean_spectrum(np.asarray(u_bi_hr), np.asarray(v_bi_hr))

    k_common = np.intersect1d(k_sr.astype(int), k_bi.astype(int)).astype(float)
    e_sr_c = np.interp(k_common, k_sr, e_sr)
    e_bi_c = np.interp(k_common, k_bi, e_bi)
    valid = np.isfinite(e_sr_c) & np.isfinite(e_bi_c) & (e_sr_c > 0) & (e_bi_c > 0)
    k_common, e_sr_c, e_bi_c = k_common[valid], e_sr_c[valid], e_bi_c[valid]

    slope_sr = slope_loglog(k_common, e_sr_c)
    slope_bi = slope_loglog(k_common, e_bi_c)
    ratio_midhi = integrated_ratio(k_common, e_sr_c, e_bi_c, args.kband_lo, args.kband_hi)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.loglog(k_common, e_bi_c, color=COLORS["cmems"], label="Bilinear-upsampled")
    ax.loglog(k_common, e_sr_c, color=COLORS["pinn"], label="PINN-SR")
    ax.set_xlabel("Wavenumber k (grid mode)")
    ax.set_ylabel("E(k)")
    ax.set_title("Isotropic Kinetic Energy Spectrum")
    ax.legend(frameon=False)
    ax.text(0.03, 0.04, f"slope_SR={slope_sr:.3f}, slope_BI={slope_bi:.3f}\nmid-high ratio(SR/BI)={ratio_midhi:.3f}", transform=ax.transAxes)
    save_fig(fig, out_dir / "fig_ke_spectrum")

    pd.DataFrame([
        {
            "slope_sr": slope_sr,
            "slope_bilinear": slope_bi,
            "mid_high_k_ratio_sr_over_bilinear": ratio_midhi,
            "kband_lo": args.kband_lo,
            "kband_hi": args.kband_hi,
            "n_spectrum_times": len(tidx),
        }
    ]).to_csv(out_dir / "spectrum_summary.csv", index=False, encoding="utf-8")

    print("[INFO] Done")
    print(f"[INFO] metrics: {out_dir / 'metrics_all.csv'}")
    print(f"[INFO] structural: {out_dir / 'metrics_structural.csv'}")
    print(f"[INFO] spectrum: {out_dir / 'spectrum_summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINN-SR 顶刊风格评估与可视化（含能谱/谱斜率）")
    parser.add_argument("--data", type=str, default=r"D:/data/output/train_data_2023.nc")
    parser.add_argument("--ckpt", type=str, default=r"D:/data/output/checkpoints_sr_mlp_strict_weighted/best_sr.pt")
    parser.add_argument("--out-dir", type=str, default=r"D:/data/output/eval_pubstyle")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--start", type=str, default="2023-02-04 00:00:00")
    parser.add_argument("--end", type=str, default="2023-03-05 23:59:59")
    parser.add_argument("--spectrum-times", type=int, default=200)
    parser.add_argument("--kband-lo", type=float, default=0.4)
    parser.add_argument("--kband-hi", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--allow-no-obs", action="store_true", help="Allow running without valid u_obs/v_obs (metrics become NaN)")
    main(parser.parse_args())
