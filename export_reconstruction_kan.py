"""Export reconstructed flow NetCDF from KAN checkpoint only."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
from pathlib import Path

import torch


def _load_export_one():
    """Load export_one from sibling file, even when cwd is not script directory."""
    try:
        from export_reconstruction_from_checkpoint import export_one as fn

        return fn
    except ModuleNotFoundError:
        module_path = Path(__file__).with_name("export_reconstruction_from_checkpoint.py")
        spec = importlib.util.spec_from_file_location("export_reconstruction_from_checkpoint", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.export_one


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="pinn_sr_kan_uv_best.pt")
    p.add_argument("--cmems", default=r"D:\data\uswc_2023_cmems_cleaned.nc")
    p.add_argument("--out", default=r"D:\data\recon_kan.nc")
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
    export_one = _load_export_one()

    kwargs = dict(
        model_type="kan",
        ckpt_path=args.ckpt,
        cmems_path=args.cmems,
        out_path=args.out,
        device=args.device,
        batch_size=args.batch_size,
        max_norm_samples=args.max_norm_samples,
        log_every_batches=args.log_every_batches,
        max_time_steps=args.max_time_steps,
        trust_checkpoint=not args.untrusted_checkpoint,
    )
    if "upscale_factor" in inspect.signature(export_one).parameters:
        kwargs["upscale_factor"] = args.upscale_factor
    export_one(**kwargs)


if __name__ == "__main__":
    main()
