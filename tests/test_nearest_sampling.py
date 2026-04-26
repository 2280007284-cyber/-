import numpy as np
import xarray as xr

from compare_kan_mlp_cmems_vs_hf import _manual_nearest_sample, _manual_nearest_sample_rect, _spatial_stride, _to_numpy_streamed
from evaluate_vs_hf_truth import _read_array_resilient
from evaluate_vs_hf_truth import _spatial_stride as _spatial_stride_eval
from evaluate_vs_hf_truth import _find as _find_eval
from evaluate_vs_hf_truth import _interp_to_hf, _interp_to_hf_chunked
from evaluate_vs_hf_truth import _init_running_stats, _accumulate_metrics, _finalize_metrics, _metrics
from evaluate_vs_hf_truth import _upscale_coord_1d
from evaluate_vs_hf_truth import _slice_src_to_hf_time
from evaluate_vs_hf_truth import _infer_bg_scale_factor
from evaluate_vs_hf_truth import _maybe_auto_unit_align


def test_manual_nearest_sample_descending_1d_grid():
    lon = np.array([2.0, 1.0, 0.0], dtype=np.float64)
    lat = np.array([11.0, 10.0], dtype=np.float64)
    u = np.array([[21.0, 11.0, 1.0], [20.0, 10.0, 0.0]], dtype=np.float64)
    v = -u
    ds = xr.Dataset({"u": (("lat", "lon"), u), "v": (("lat", "lon"), v)}, coords={"lon": lon, "lat": lat})

    lon_q, lat_q = np.meshgrid(np.array([0.2, 1.8]), np.array([10.1, 10.9]))
    u_q, v_q = _manual_nearest_sample(ds, "u", "v", "lon", "lat", lon_q, lat_q)

    exp_u = np.array([[0.0, 20.0], [1.0, 21.0]])
    assert np.allclose(u_q, exp_u)
    assert np.allclose(v_q, -exp_u)


def test_manual_nearest_sample_curvilinear_grid():
    lon2d = np.array([[0.0, 1.2], [0.1, 1.1]], dtype=np.float64)
    lat2d = np.array([[10.0, 10.0], [11.0, 11.0]], dtype=np.float64)
    u = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    v = 10.0 + u
    ds = xr.Dataset(
        {"u": (("y", "x"), u), "v": (("y", "x"), v)},
        coords={"longitude": (("y", "x"), lon2d), "latitude": (("y", "x"), lat2d)},
    )

    lon_q = np.array([[0.05, 1.15]], dtype=np.float64)
    lat_q = np.array([[10.95, 10.05]], dtype=np.float64)
    u_q, v_q = _manual_nearest_sample(ds, "u", "v", "longitude", "latitude", lon_q, lat_q)

    assert np.allclose(u_q, np.array([[3.0, 2.0]]))
    assert np.allclose(v_q, np.array([[13.0, 12.0]]))


def test_manual_nearest_sample_rect_uses_1d_queries():
    lon = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    lat = np.array([10.0, 11.0], dtype=np.float64)
    u = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    v = -u
    ds = xr.Dataset({"u": (("lat", "lon"), u), "v": (("lat", "lon"), v)}, coords={"lon": lon, "lat": lat})

    u_q, v_q = _manual_nearest_sample_rect(ds, "u", "v", "lon", "lat", np.array([0.2, 1.9]), np.array([10.2, 10.8]))
    assert np.allclose(u_q, np.array([[1.0, 3.0], [4.0, 6.0]]))
    assert np.allclose(v_q, -u_q)


def test_spatial_stride_reduces_grid_size():
    ds = xr.Dataset(
        {"u": (("time", "lat", "lon"), np.zeros((1, 6, 8), dtype=np.float32))},
        coords={"time": [0], "lat": np.arange(6), "lon": np.arange(8)},
    )
    ds2 = _spatial_stride(ds, 2)
    assert ds2.sizes["lat"] == 3
    assert ds2.sizes["lon"] == 4


def test_to_numpy_streamed_matches_values():
    da = xr.DataArray(np.arange(2 * 5 * 7, dtype=np.int16).reshape(2, 5, 7), dims=("time", "lat", "lon"))
    got = _to_numpy_streamed(da, chunk_len=2)
    assert got.dtype == np.int16
    assert np.array_equal(got, da.values)


def test_read_array_resilient_on_in_memory_dataset():
    ds = xr.Dataset({"u": (("time", "lat", "lon"), np.arange(12, dtype=np.int16).reshape(1, 3, 4))})
    got = _read_array_resilient(ds, "u", chunk_len=2)
    assert np.array_equal(got, ds["u"].values)


def test_eval_spatial_stride_reduces_grid_size():
    ds = xr.Dataset(
        {"u": (("time", "lat", "lon"), np.zeros((1, 6, 8), dtype=np.float32))},
        coords={"time": [0], "lat": np.arange(6), "lon": np.arange(8)},
    )
    ds2 = _spatial_stride_eval(ds, 2)
    assert ds2.sizes["lat"] == 3
    assert ds2.sizes["lon"] == 4


def test_eval_find_prefers_data_var_over_coord():
    ds = xr.Dataset(
        data_vars={"u": (("y", "x"), np.ones((2, 2), dtype=np.float32))},
        coords={"lon": ("x", [0.0, 1.0]), "u_coord": ("y", [1.0, 2.0])},
    )
    assert _find_eval(ds, ["u", "u_coord"], "u") == "u"


def test_interp_to_hf_chunked_matches_full():
    t = np.array(["2023-01-01"], dtype="datetime64[ns]")
    lat = np.array([30.0, 31.0, 32.0, 33.0], dtype=np.float64)
    lon = np.array([-120.0, -119.0, -118.0], dtype=np.float64)
    u = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
    v = -u
    src = xr.Dataset({"u_recon": (("time", "lat", "lon"), u), "v_recon": (("time", "lat", "lon"), v)}, coords={"time": t, "lat": lat, "lon": lon})
    hf = xr.Dataset({"u": (("time", "lat", "lon"), u), "v": (("time", "lat", "lon"), v)}, coords={"time": t, "lat": lat, "lon": lon})

    up, vp, ut, vt = _interp_to_hf(src, hf, "u_recon", "v_recon", interp_method="nearest")
    upc, vpc, utc, vtc = _interp_to_hf_chunked(src, hf, "u_recon", "v_recon", interp_method="nearest", lat_chunk=2)
    assert np.allclose(up.ravel(), upc)
    assert np.allclose(vp.ravel(), vpc)
    assert np.allclose(ut.ravel(), utc)
    assert np.allclose(vt.ravel(), vtc)


def test_interp_to_hf_respects_max_time_gap():
    t_src = np.array(["2023-01-01T00:00:00"], dtype="datetime64[ns]")
    t_hf = np.array(["2023-01-03T00:00:00"], dtype="datetime64[ns]")
    lat = np.array([30.0], dtype=np.float64)
    lon = np.array([-120.0], dtype=np.float64)
    src = xr.Dataset({"u_recon": (("time", "lat", "lon"), np.zeros((1, 1, 1), dtype=np.float32)),
                      "v_recon": (("time", "lat", "lon"), np.zeros((1, 1, 1), dtype=np.float32))},
                     coords={"time": t_src, "lat": lat, "lon": lon})
    hf = xr.Dataset({"u": (("time", "lat", "lon"), np.zeros((1, 1, 1), dtype=np.float32)),
                     "v": (("time", "lat", "lon"), np.zeros((1, 1, 1), dtype=np.float32))},
                    coords={"time": t_hf, "lat": lat, "lon": lon})
    try:
        _interp_to_hf(src, hf, "u_recon", "v_recon", interp_method="nearest", max_time_gap_hours=1.0)
        assert False, "expected RuntimeError for excessive time gap"
    except RuntimeError as exc:
        assert "time gap too large" in str(exc)


def test_running_metrics_matches_batch_metrics():
    u_pred = np.array([0.1, 0.2, 0.3, np.nan], dtype=np.float64)
    v_pred = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    u_true = np.array([0.1, 0.25, 0.35, 0.0], dtype=np.float64)
    v_true = np.array([0.0, 0.15, 0.1, 0.2], dtype=np.float64)

    m_batch = _metrics(u_pred, v_pred, u_true, v_true, max_abs_speed=5.0)
    st = _init_running_stats()
    _accumulate_metrics(st, u_pred[:2], v_pred[:2], u_true[:2], v_true[:2], max_abs_speed=5.0)
    _accumulate_metrics(st, u_pred[2:], v_pred[2:], u_true[2:], v_true[2:], max_abs_speed=5.0)
    m_run = _finalize_metrics(st)

    assert m_batch["n"] == m_run["n"]
    assert np.allclose(m_batch["rmse_u"], m_run["rmse_u"])
    assert np.allclose(m_batch["rmse_v"], m_run["rmse_v"])


def test_upscale_coord_1d_preserves_direction():
    c = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    cu = _upscale_coord_1d(c, scale=2)
    assert cu.shape[0] == 6
    assert cu[0] > cu[-1]


def test_slice_src_to_hf_time_limits_window():
    t = np.array(["2023-06-01", "2023-07-10", "2023-08-01"], dtype="datetime64[ns]")
    ds = xr.Dataset({"u_bg": (("time",), np.array([1, 2, 3], dtype=np.float32))}, coords={"time": t})
    out = _slice_src_to_hf_time(ds, pad_hours=0, hf_time_range=(np.datetime64("2023-07-01"), np.datetime64("2023-07-31")))
    assert out.sizes["time"] == 1


def test_infer_bg_scale_factor_matches_training_heuristic():
    u = np.array([[[250.0]]], dtype=np.float32)
    v = np.array([[[260.0]]], dtype=np.float32)
    assert _infer_bg_scale_factor(u, v) == 1000.0


def test_auto_unit_align_can_detect_100x_scale():
    up = np.array([100.0, 200.0], dtype=np.float32)
    vp = np.array([0.0, 0.0], dtype=np.float32)
    ut = np.array([1.0, 2.0], dtype=np.float32)
    vt = np.array([0.0, 0.0], dtype=np.float32)
    up2, vp2, fac = _maybe_auto_unit_align(up, vp, ut, vt, pred_units_hint="", true_units_hint="m/s", enabled=True)
    assert fac == 0.01
    assert np.allclose(up2, ut)
    assert np.allclose(vp2, vt)
