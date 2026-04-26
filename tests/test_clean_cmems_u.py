import numpy as np
import xarray as xr

from clean_cmems_u import clean_cmems_u


def test_clean_cmems_u_fills_and_clips(tmp_path):
    t = np.array(['2023-01-01T00:00:00','2023-01-01T06:00:00','2023-01-01T12:00:00'], dtype='datetime64[ns]')
    lat = np.array([30.0, 31.0])
    lon = np.array([-120.0, -119.0])

    u = np.array(
        [
            [[0.1, np.nan], [0.2, 9.9]],
            [[0.2, 0.3], [np.nan, 0.1]],
            [[0.4, 0.2], [0.1, 0.0]],
        ],
        dtype=np.float32,
    )
    v = np.array(
        [
            [[-0.1, -0.2], [np.nan, -8.8]],
            [[-0.2, -0.3], [-0.1, np.nan]],
            [[-0.4, -0.2], [-0.1, 0.0]],
        ],
        dtype=np.float32,
    )

    ds = xr.Dataset(
        data_vars={
            'u': (('time', 'lat', 'lon'), u),
            'v': (('time', 'lat', 'lon'), v),
        },
        coords={'time': t, 'lat': lat, 'lon': lon},
    )

    in_path = tmp_path / 'in.nc'
    out_path = tmp_path / 'out.nc'
    ds.to_netcdf(in_path)

    clean_cmems_u(
        in_path=str(in_path),
        out_path=str(out_path),
        max_abs_speed=3.0,
        interp_limit=8,
        smooth_window=1,
    )

    out = xr.open_dataset(out_path)
    assert 'u_clean' in out and 'v_clean' in out
    assert out['u_clean'].dtype == np.float32
    assert out['v_clean'].dtype == np.float32

    assert np.isfinite(out['u_clean'].values).all()
    assert np.isfinite(out['v_clean'].values).all()
    assert np.nanmax(np.abs(out['u_clean'].values)) <= 3.0
    assert np.nanmax(np.abs(out['v_clean'].values)) <= 3.0
