import numpy as np
from scipy.sparse import csc_matrix

import pyatb.tb.solver as solver_module
from pyatb.io.default_input import INPUT


class _DummyInterfacePython:
    def __init__(self, *args, **kwargs):
        pass

    def set_HSR(self, *args, **kwargs):
        return None

    def set_HSR_sparse(self, *args, **kwargs):
        return None


def _diagonal_triu_matrix(diagonal):
    basis_num = len(diagonal)
    triu_size = basis_num * (basis_num + 1) // 2
    data = np.zeros((1, triu_size), dtype=complex)
    offset = 0
    for row, value in enumerate(diagonal):
        data[0, offset] = value
        offset += basis_num - row
    return csc_matrix(data)


def test_band_structure_defaults_include_sparse_shift_invert_options():
    band_structure = INPUT["BAND_STRUCTURE"]

    assert band_structure["solver"][-1] == "dense"
    assert band_structure["fermi_band_num"][-1] == 0


def test_sparse_shift_invert_solver_matches_generalized_dense_reference(monkeypatch):
    monkeypatch.setattr(solver_module, "tb_solver_", _DummyInterfacePython)

    tb_solver = solver_module.solver(1.0, np.eye(3))
    r_direct_coor = np.array([[0, 0, 0]], dtype=int)
    hr = _diagonal_triu_matrix([-1.0, -0.2, 0.6, 8.0])
    sr = _diagonal_triu_matrix([1.0, 1.0, 2.0, 4.0])
    tb_solver.set_HSR_sparse(1, r_direct_coor, 4, hr, sr)

    eigenvectors, eigenvalues = tb_solver.diago_H_near_fermi(
        np.array([[0.0, 0.0, 0.0]], dtype=float),
        fermi_energy=0.0,
        band_num=2,
    )
    eigenvalues_only = tb_solver.diago_H_eigenvaluesOnly_near_fermi(
        np.array([[0.0, 0.0, 0.0]], dtype=float),
        fermi_energy=0.0,
        band_num=2,
    )

    np.testing.assert_allclose(eigenvalues[0], np.array([-0.2, 0.3]))
    np.testing.assert_allclose(eigenvalues_only[0], np.array([-0.2, 0.3]))
    hk = tb_solver.get_Hk_sparse(np.array([0.0, 0.0, 0.0], dtype=float)).toarray()
    sk = tb_solver.get_Sk_sparse(np.array([0.0, 0.0, 0.0], dtype=float)).toarray()
    for iband, eigenvalue in enumerate(eigenvalues[0]):
        vector = eigenvectors[0, :, iband]
        np.testing.assert_allclose(hk @ vector, eigenvalue * (sk @ vector), atol=1e-10)
