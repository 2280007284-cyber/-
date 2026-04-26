import numpy as np

from compare_kan_mlp_cmems_vs_hf import _scale_to_mps as scale_hf
from compare_kan_mlp_vs_cmems import _scale_to_mps as scale_cmems


def test_unknown_units_do_not_auto_scale():
    x = np.array([0.02, 0.05, 0.1], dtype=np.float64)
    y_hf, f_hf = scale_hf(x, units_hint="")
    y_cm, f_cm = scale_cmems(x, units_hint="")
    assert f_hf == 1.0
    assert f_cm == 1.0
    assert np.allclose(y_hf, x)
    assert np.allclose(y_cm, x)


def test_known_units_convert_to_mps():
    x = np.array([10.0, 50.0], dtype=np.float64)
    y_hf, f_hf = scale_hf(x, units_hint="cm/s")
    y_cm, f_cm = scale_cmems(x, units_hint="cm s-1")
    assert f_hf == 0.01
    assert f_cm == 0.01
    assert np.allclose(y_hf, [0.1, 0.5])
    assert np.allclose(y_cm, [0.1, 0.5])
