import numpy as np
from types import SimpleNamespace

from pyatb.fermi.band_structure import Band_Structure
from pyatb.io.default_input import INPUT
from pyatb.tb.solver import solver


def test_band_structure_registers_parpack_defaults():
    config = INPUT["BAND_STRUCTURE"]

    assert config["eigensolver"][-1] == "parpack"
    assert config["parpack_nev"][-1] == 50
    assert config["parpack_ncv"][-1] == 0
    assert config["parpack_tol"][-1] == 0.0
    assert config["parpack_maxiter"][-1] == 300


def test_configure_eigensolver_uses_parpack_settings():
    band = Band_Structure.__new__(Band_Structure)
    band._Band_Structure__tb = SimpleNamespace(basis_num=64)

    band._Band_Structure__configure_eigensolver(
        fermi_energy=1.25,
        band_range=np.array([-1, -1], dtype=int),
        eigensolver="parpack",
        arpack_nev=20,
        arpack_ncv=0,
        arpack_tol=1e-6,
        arpack_maxiter=200,
        parpack_nev=12,
        parpack_ncv=30,
        parpack_tol=1e-8,
        parpack_maxiter=99,
    )

    assert band._Band_Structure__eigensolver == "parpack"
    assert band._Band_Structure__iterative_nev == 12
    assert band._Band_Structure__iterative_ncv == 30
    assert band._Band_Structure__iterative_sigma == 1.25
    assert band._Band_Structure__iterative_tol == 1e-8
    assert band._Band_Structure__iterative_maxiter == 99
    np.testing.assert_array_equal(band.band_range, np.array([1, 12], dtype=int))
    assert band.cal_all_band is False


def test_configure_eigensolver_falls_back_to_dense_for_full_spectrum_request():
    band = Band_Structure.__new__(Band_Structure)
    band._Band_Structure__tb = SimpleNamespace(basis_num=10)

    band._Band_Structure__configure_eigensolver(
        fermi_energy=0.0,
        band_range=np.array([-1, -1], dtype=int),
        eigensolver="parpack",
        parpack_nev=10,
    )

    assert band._Band_Structure__eigensolver == "lapack"
    assert band.cal_all_band is True
    np.testing.assert_array_equal(band.band_range, np.array([1, 10], dtype=int))


def test_solver_parpack_passes_fortran_comm_handle():
    class FakeComm:
        def py2f(self):
            return 42

    class FakeBackend:
        def diago_H_parpack(self, kpoints, nev, sigma, ncv, tol, maxiter, comm_f, eigenvectors, eigenvalues):
            assert comm_f == 42
            eigenvalues[:] = sigma

        def diago_H_eigenvaluesOnly_parpack(self, kpoints, nev, sigma, ncv, tol, maxiter, comm_f, eigenvalues):
            assert comm_f == 42
            eigenvalues[:] = sigma

    tb_solver = solver.__new__(solver)
    tb_solver.basis_num = 3
    tb_solver.tb_solver = FakeBackend()
    kpoints = np.zeros((2, 3), dtype=float)

    vectors, values = tb_solver.diago_H_parpack(kpoints, nev=2, sigma=3.5, comm=FakeComm())
    values_only = tb_solver.diago_H_eigenvaluesOnly_parpack(kpoints, nev=2, sigma=1.5, comm=FakeComm())

    assert vectors.shape == (2, 3, 2)
    np.testing.assert_allclose(values, np.full((2, 2), 3.5))
    np.testing.assert_allclose(values_only, np.full((2, 2), 1.5))
