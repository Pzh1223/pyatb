import numpy as np

from pyatb.fermi.cohp import cohp_values_one_k


def test_cohp_values_one_k_matches_standalone_formula():
    matrix = np.array(
        [
            [1.0, 2.0 + 0.5j, 0.0],
            [2.0 - 0.5j, 3.0, 4.0],
            [0.0, 4.0, 5.0],
        ],
        dtype=np.complex128,
    )
    eigenvalues = np.array([-1.0, 0.5], dtype=float)
    eigenvectors = np.array(
        [
            [1.0 + 0.0j, 0.0 + 1.0j],
            [0.5 + 0.5j, 1.0 + 0.0j],
            [0.25 + 0.0j, 0.5 - 0.25j],
        ],
        dtype=np.complex128,
    )

    energies, values = cohp_values_one_k(matrix, eigenvalues, eigenvectors, [0, 1], [2])

    expected = []
    for ib in range(eigenvectors.shape[1]):
        total = 0.0
        for iorb in [0, 1]:
            total += (eigenvectors[iorb, ib].conjugate() * matrix[iorb, 2] * eigenvectors[2, ib]).real
        expected.append(total)

    np.testing.assert_allclose(energies, eigenvalues)
    np.testing.assert_allclose(values, np.array(expected))
