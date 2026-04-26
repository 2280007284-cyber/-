"""PINN-SR workflow for reconstructing high-resolution ocean currents.

This module provides a minimal, executable template to:
1. Load CMEMS gridded currents (background field).
2. Load GDP drifter trajectories corrected with ERA5 wind-slip and Stokes drift.
3. Train a PINN-SR model with data + physics losses.
4. Infer high-resolution currents on a target grid.

The implementation is intentionally compact so you can adapt variable names,
products, and equation terms to your specific region/time span.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class GridSpec:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    t_min: float
    t_max: float
    nx: int
    ny: int
    nt: int


@dataclass
class TrainConfig:
    epochs: int = 4000
    batch_size: int = 4096
    lr: float = 1e-3
    lambda_data: float = 1.0
    lambda_bg: float = 0.5
    lambda_phy: float = 0.1
    hidden_dim: int = 128
    depth: int = 6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, scale: float = 5.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dim, out_dim) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = 2 * torch.pi * x @ self.B
        return torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)


class PINNSR(nn.Module):
    """Predicts (u, v, psi) from (lon, lat, t)."""

    def __init__(self, hidden_dim: int = 128, depth: int = 6, ff_dim: int = 32):
        super().__init__()
        self.ff = FourierFeatures(3, ff_dim)
        in_dim = ff_dim * 2
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        z = self.ff(xyt)
        return self.net(z)


def load_cmems_background() -> Dict[str, np.ndarray]:
    """Replace with xarray/netCDF loader. Expected flattened samples.

    Returns keys: xyt (N,3), uv (N,2)
    """
    raise NotImplementedError("Use xarray.open_dataset and interpolate to training grid.")


def load_gdp_corrected() -> Dict[str, np.ndarray]:
    """Replace with your corrected GDP trajectories.

    GDP should already include ERA5 wind-slip correction and Stokes correction.
    Returns keys: xyt (M,3), uv (M,2)
    """
    raise NotImplementedError("Provide corrected drifter records with lon/lat/time/u/v.")


def physics_residual(model: nn.Module, xyt: torch.Tensor) -> torch.Tensor:
    """Simple incompressibility residual: du/dx + dv/dy = 0.

    Extend this with momentum equations if needed.
    """
    xyt.requires_grad_(True)
    pred = model(xyt)
    u = pred[:, 0:1]
    v = pred[:, 1:2]

    grads_u = torch.autograd.grad(u.sum(), xyt, create_graph=True)[0]
    grads_v = torch.autograd.grad(v.sum(), xyt, create_graph=True)[0]
    div = grads_u[:, 0:1] + grads_v[:, 1:2]
    return torch.mean(div**2)


def train_pinn_sr(
    model: PINNSR,
    cmems: Dict[str, np.ndarray],
    gdp: Dict[str, np.ndarray],
    cfg: TrainConfig,
) -> None:
    device = torch.device(cfg.device)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    cmems_xyt = torch.tensor(cmems["xyt"], dtype=torch.float32, device=device)
    cmems_uv = torch.tensor(cmems["uv"], dtype=torch.float32, device=device)
    gdp_xyt = torch.tensor(gdp["xyt"], dtype=torch.float32, device=device)
    gdp_uv = torch.tensor(gdp["uv"], dtype=torch.float32, device=device)

    for epoch in range(1, cfg.epochs + 1):
        idx_c = torch.randint(0, cmems_xyt.shape[0], (cfg.batch_size,), device=device)
        idx_g = torch.randint(0, gdp_xyt.shape[0], (cfg.batch_size,), device=device)

        x_c, y_c = cmems_xyt[idx_c], cmems_uv[idx_c]
        x_g, y_g = gdp_xyt[idx_g], gdp_uv[idx_g]

        pred_c = model(x_c)[:, :2]
        pred_g = model(x_g)[:, :2]

        loss_bg = mse(pred_c, y_c)
        loss_data = mse(pred_g, y_g)

        x_phy = torch.cat([x_c[: cfg.batch_size // 2], x_g[: cfg.batch_size // 2]], dim=0)
        loss_phy = physics_residual(model, x_phy)

        loss = cfg.lambda_bg * loss_bg + cfg.lambda_data * loss_data + cfg.lambda_phy * loss_phy

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 200 == 0:
            print(
                f"epoch={epoch:5d} total={loss.item():.6f} "
                f"data={loss_data.item():.6f} bg={loss_bg.item():.6f} phy={loss_phy.item():.6f}"
            )


def make_highres_grid(spec: GridSpec) -> np.ndarray:
    lon = np.linspace(spec.lon_min, spec.lon_max, spec.nx)
    lat = np.linspace(spec.lat_min, spec.lat_max, spec.ny)
    tim = np.linspace(spec.t_min, spec.t_max, spec.nt)
    ll, aa, tt = np.meshgrid(lon, lat, tim, indexing="xy")
    grid = np.column_stack([ll.ravel(), aa.ravel(), tt.ravel()])
    return grid


def infer_highres(model: PINNSR, grid_xyt: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(grid_xyt, dtype=torch.float32, device=device)
        pred = model(x).cpu().numpy()
    uv = pred[:, :2]
    psi = pred[:, 2:3]
    return uv, psi


def main() -> None:
    """Example entrypoint.

    Replace load_* functions before running.
    """
    cfg = TrainConfig()
    model = PINNSR(hidden_dim=cfg.hidden_dim, depth=cfg.depth)

    cmems = load_cmems_background()
    gdp = load_gdp_corrected()
    train_pinn_sr(model, cmems, gdp, cfg)

    grid = make_highres_grid(
        GridSpec(
            lon_min=120.0,
            lon_max=130.0,
            lat_min=20.0,
            lat_max=30.0,
            t_min=0.0,
            t_max=30.0,
            nx=400,
            ny=400,
            nt=24,
        )
    )
    uv, psi = infer_highres(model, grid, cfg.device)
    print("High-res output shape (u,v):", uv.shape)
    print("Streamfunction shape:", psi.shape)


if __name__ == "__main__":
    main()
