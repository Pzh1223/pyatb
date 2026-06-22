import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from pyatb.interface_python import interface_python as tb_solver_

IMAG_EIGENVALUE_TOL = 1e-8
OVERLAP_SINGULAR_TOL = 1e-12
MIN_SPARSE_BASIS = 4
NEAR_FULL_BAND_MARGIN = 1


class solver:
    def __init__(self, lattice_constant, lattice_vector):
        self.tb_solver = tb_solver_(lattice_constant, lattice_vector)
        self._sparse_enabled = False
        self._sparse_hr = None
        self._sparse_sr = None
        self._triu_rows = None
        self._triu_cols = None

    def set_HSR(self, R_num, R_direct_coor, basis_num, HR, SR):
        self.R_num = R_num
        self.basis_num = basis_num
        self.R_direct_coor = np.asarray(R_direct_coor, dtype=float)
        self._sparse_enabled = False
        self._sparse_hr = None
        self._sparse_sr = None
        self.tb_solver.set_HSR(R_num, R_direct_coor, basis_num, HR, SR)

    def set_HSR_sparse(self, R_num, R_direct_coor, basis_num, HR, SR):
        self.R_num = R_num
        self.basis_num = basis_num
        self.R_direct_coor = np.asarray(R_direct_coor, dtype=float)
        self._sparse_enabled = True
        self._sparse_hr = HR.copy()
        self._sparse_sr = SR.copy()
        self._triu_rows, self._triu_cols = np.triu_indices(self.basis_num)
        self.tb_solver.set_HSR_sparse(R_num, R_direct_coor, basis_num, HR, SR)

    def set_rR(self, rR_x, rR_y, rR_z):
        self.tb_solver.set_rR(rR_x, rR_y, rR_z)

    def set_rR_sparse(self, rR_x, rR_y, rR_z):
        self.tb_solver.set_rR_sparse(rR_x, rR_y, rR_z)

    def set_single_atom_position(self, atom_label, na, tau_car):
        self.tb_solver.set_single_atom_position(atom_label, na, tau_car)

    def set_single_atom_orb(self, atom_index, nwl, l_nchi, mesh, dr, numerical_orb):
        self.tb_solver.set_single_atom_orb(atom_index, nwl, l_nchi, mesh, dr, numerical_orb)

    def get_Hk(self, k_direct_coor):
        kpoint_num = k_direct_coor.shape[0]
        Hk = np.zeros([kpoint_num, self.basis_num, self.basis_num], dtype=complex)
        self.tb_solver.get_Hk(k_direct_coor, Hk)

        return Hk

    def get_Sk(self, k_direct_coor):
        kpoint_num = k_direct_coor.shape[0]
        Sk = np.zeros([kpoint_num, self.basis_num, self.basis_num], dtype=complex)
        self.tb_solver.get_Sk(k_direct_coor, Sk)

        return Sk
    
    def get_rk(self, k_direct_coor):
        kpoint_num = k_direct_coor.shape[0]
        rk = np.zeros([kpoint_num, 3, self.basis_num, self.basis_num], dtype=complex)
        self.tb_solver.get_rk(k_direct_coor, rk)

        return rk
    
    def get_partial_Hk(self, k_direct_coor):
        kpoint_num = k_direct_coor.shape[0]
        partial_Hk = np.zeros([kpoint_num, 3, self.basis_num, self.basis_num], dtype=complex)
        self.tb_solver.get_partial_Hk(k_direct_coor, partial_Hk)

        return partial_Hk
    
    def get_partial_Sk(self, k_direct_coor):
        kpoint_num = k_direct_coor.shape[0]
        partial_Sk = np.zeros([kpoint_num, 3, self.basis_num, self.basis_num], dtype=complex)
        self.tb_solver.get_partial_Sk(k_direct_coor, partial_Sk)

        return partial_Sk
    
    def get_surface_Hk00_and_Hk01(self, direction, coupling_layers, k_direct_coor):
        kpoint_num = k_direct_coor.shape[0]
        matrix_dim = coupling_layers * self.basis_num
        Hk00 = np.zeros([kpoint_num, matrix_dim, matrix_dim], dtype=complex)
        Hk01 = np.zeros([kpoint_num, matrix_dim, matrix_dim], dtype=complex)
        self.tb_solver.get_surface_Hk00_and_Hk01(direction, coupling_layers, k_direct_coor, Hk00, Hk01)

        return Hk00, Hk01
    
    def get_surface_Sk00_and_Sk01(self, direction, coupling_layers, k_direct_coor):
        kpoint_num = k_direct_coor.shape[0]
        matrix_dim = coupling_layers * self.basis_num
        Sk00 = np.zeros([kpoint_num, matrix_dim, matrix_dim], dtype=complex)
        Sk01 = np.zeros([kpoint_num, matrix_dim, matrix_dim], dtype=complex)
        self.tb_solver.get_surface_Sk00_and_Sk01(direction, coupling_layers, k_direct_coor, Sk00, Sk01)

        return Sk00, Sk01
    
    def get_surface_G00(self, omega, Hk00, Hk01, Sk00, Sk01):
        """
        omega为复数能量(eV)
        Hk00, Hk01, Sk00, Sk01均为指定k点的二维矩阵
        """
        matrix_dim = Hk00.shape[0]
        G00_top = np.zeros([matrix_dim, matrix_dim], dtype=complex)
        G00_bottom = np.zeros([matrix_dim, matrix_dim], dtype=complex)
        G00_bulk = np.zeros([matrix_dim, matrix_dim], dtype=complex)
        self.tb_solver.get_surface_G00(omega, Hk00, Hk01, Sk00, Sk01, G00_top, G00_bottom, G00_bulk)

        return G00_top, G00_bottom, G00_bulk

    
    def get_surface_spectral_fun_by_green(self, direction, coupling_layers, omega_num, domega, start_omega, eta, iter_max, converged_eps, k_direct_coor):
        kpoint_num = k_direct_coor.shape[0]
        spectral_fun = np.zeros([kpoint_num, omega_num])
        self.tb_solver.get_surface_spectral_fun_by_green(direction, coupling_layers, omega_num, domega, start_omega, eta, iter_max, converged_eps, k_direct_coor, spectral_fun)
        return spectral_fun
    
    def get_surface_spectral_fun_by_green_top_bottom_bulk(self, direction, coupling_layers, omega_num, domega, start_omega, eta, iter_max, converged_eps, k_direct_coor):
        kpoint_num = k_direct_coor.shape[0]
        spectral_fun_top = np.zeros([kpoint_num, omega_num])
        spectral_fun_bottom = np.zeros([kpoint_num, omega_num])
        spectral_fun_bulk = np.zeros([kpoint_num, omega_num])
        self.tb_solver.get_surface_spectral_fun_by_green_top_bottom_bulk(direction, coupling_layers, omega_num, domega, start_omega, eta, iter_max, converged_eps, k_direct_coor, spectral_fun_top, spectral_fun_bottom, spectral_fun_bulk)
        return spectral_fun_top, spectral_fun_bottom, spectral_fun_bulk

    def get_surface_spectral_fun_by_Tmatrix(self, direction, coupling_layers, calculate_layer, omega_num, domega, start_omega, eta, iter_max, converged_eps, k_direct_coor):
        kpoint_num = k_direct_coor.shape[0]
        spect_matrix_l = np.zeros([kpoint_num, calculate_layer, omega_num])
        spect_matrix_r = np.zeros([kpoint_num, calculate_layer, omega_num])
        self.tb_solver.get_surface_spectral_fun_by_Tmatrix(direction, coupling_layers, calculate_layer, omega_num, domega, start_omega, eta, iter_max, converged_eps, k_direct_coor, spect_matrix_l, spect_matrix_r)
        return spect_matrix_l, spect_matrix_r

    def diago_H(self, k_direct_coor):
        kpoint_num = k_direct_coor.shape[0]
        eigenvectors = np.zeros([kpoint_num, self.basis_num, self.basis_num], dtype=complex)
        eigenvalues = np.zeros([kpoint_num, self.basis_num], dtype=float)
        self.tb_solver.diago_H(k_direct_coor, eigenvectors, eigenvalues)

        return eigenvectors, eigenvalues
    
    def diago_H_range(self, k_direct_coor, lower_band_index, upper_band_index):
        """
        lower_band_index and upper_band_index is counting from 1.
        upper_band_index >= lower_band_index
        """
        if lower_band_index <= 0:
            raise ValueError("lower_band_index is counting from 1.")
        
        if lower_band_index > upper_band_index:
            raise ValueError("lower_band_index need small than upper_band_index.")
        
        kpoint_num = k_direct_coor.shape[0]
        cal_band_num = upper_band_index - lower_band_index + 1
        eigenvectors = np.zeros([kpoint_num, self.basis_num, cal_band_num], dtype=complex)
        eigenvalues = np.zeros([kpoint_num, cal_band_num], dtype=float)
        self.tb_solver.diago_H_range(k_direct_coor, lower_band_index, upper_band_index, eigenvectors, eigenvalues)

        return eigenvectors, eigenvalues

    def diago_H_eigenvaluesOnly(self, k_direct_coor):
        kpoint_num = k_direct_coor.shape[0]
        eigenvalues = np.zeros([kpoint_num, self.basis_num], dtype=float)
        self.tb_solver.diago_H_eigenvaluesOnly(k_direct_coor, eigenvalues)

        return eigenvalues
    
    def diago_H_eigenvaluesOnly_range(self, k_direct_coor, lower_band_index, upper_band_index):
        """
        lower_band_index and upper_band_index is counting from 1.
        upper_band_index >= lower_band_index
        """
        if lower_band_index <= 0:
            raise ValueError("lower_band_index is counting from 1.")
        
        if lower_band_index > upper_band_index:
            raise ValueError("lower_band_index need small than upper_band_index.")
        
        kpoint_num = k_direct_coor.shape[0]
        cal_band_num = upper_band_index - lower_band_index + 1
        eigenvalues = np.zeros([kpoint_num, cal_band_num], dtype=float)
        self.tb_solver.diago_H_eigenvaluesOnly_range(k_direct_coor, lower_band_index, upper_band_index, eigenvalues)

        return eigenvalues

    def _build_sparse_matrix(self, XR, phase):
        if not self._sparse_enabled:
            raise ValueError("Sparse solver is only available after set_HSR_sparse().")

        upper_triangle = np.asarray(XR.transpose().dot(phase)).reshape(-1)
        matrix_upper = coo_matrix(
            (upper_triangle, (self._triu_rows, self._triu_cols)),
            shape=(self.basis_num, self.basis_num),
        ).tocsc()
        matrix = matrix_upper + matrix_upper.getH()
        diagonal = matrix_upper.diagonal()
        if diagonal.size:
            matrix = matrix - coo_matrix(
                (diagonal, (np.arange(self.basis_num), np.arange(self.basis_num))),
                shape=(self.basis_num, self.basis_num),
            ).tocsc()
        return matrix

    def _get_sparse_phase(self, k_direct_coor):
        return np.exp(2j * np.pi * (self.R_direct_coor @ k_direct_coor))

    def get_Hk_sparse(self, k_direct_coor):
        phase = self._get_sparse_phase(np.asarray(k_direct_coor, dtype=float))
        return self._build_sparse_matrix(self._sparse_hr, phase)

    def get_Sk_sparse(self, k_direct_coor):
        phase = self._get_sparse_phase(np.asarray(k_direct_coor, dtype=float))
        return self._build_sparse_matrix(self._sparse_sr, phase)

    def get_HSk_sparse(self, k_direct_coor):
        phase = self._get_sparse_phase(np.asarray(k_direct_coor, dtype=float))
        return (
            self._build_sparse_matrix(self._sparse_hr, phase),
            self._build_sparse_matrix(self._sparse_sr, phase),
        )

    def _dense_near_sigma(self, Hk_sparse, Sk_sparse, sigma, band_num, return_vectors):
        eigenvalues, eigenvectors = eigh(Hk_sparse.toarray(), Sk_sparse.toarray())
        order = np.argsort(np.abs(eigenvalues - sigma))[:band_num]
        eigenvalues = np.asarray(eigenvalues[order], dtype=float)
        sort_index = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_index]
        if not return_vectors:
            return eigenvalues

        eigenvectors = np.asarray(eigenvectors[:, order], dtype=complex)[:, sort_index]
        return eigenvectors, eigenvalues

    def _as_real_eigenvalues(self, eigenvalues):
        eigenvalues = np.asarray(eigenvalues)
        if np.abs(eigenvalues.imag).max() > IMAG_EIGENVALUE_TOL:
            raise ValueError("Sparse solver returned eigenvalues with non-negligible imaginary parts.")
        return np.asarray(np.real(eigenvalues), dtype=float)

    def _solve_sparse_near_sigma(self, Hk_sparse, Sk_sparse, sigma, band_num, return_vectors):
        if band_num <= 0:
            raise ValueError("band_num must be positive for sparse near-Fermi solving.")
        if band_num > self.basis_num:
            raise ValueError("band_num cannot exceed basis_num.")

        if self.basis_num == 1:
            overlap = Sk_sparse[0, 0].real
            if abs(overlap) <= OVERLAP_SINGULAR_TOL:
                raise ValueError("Sparse solver requires a non-singular overlap matrix.")
            eigenvalues = np.array([Hk_sparse[0, 0].real / overlap], dtype=float)
            if not return_vectors:
                return eigenvalues
            return np.array([[1.0 + 0.0j]], dtype=complex), eigenvalues

        if band_num >= self.basis_num:
            return self._dense_near_sigma(Hk_sparse, Sk_sparse, sigma, band_num, return_vectors)

        if self.basis_num < MIN_SPARSE_BASIS or band_num >= self.basis_num - NEAR_FULL_BAND_MARGIN:
            return self._dense_near_sigma(Hk_sparse, Sk_sparse, sigma, band_num, return_vectors)

        result = eigsh(
            Hk_sparse,
            k=band_num,
            M=Sk_sparse,
            sigma=sigma,
            which='LM',
            return_eigenvectors=return_vectors,
        )

        if return_vectors:
            eigenvalues, eigenvectors = result
            eigenvalues = self._as_real_eigenvalues(eigenvalues)
            sort_index = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[sort_index]
            eigenvectors = np.asarray(eigenvectors[:, sort_index], dtype=complex)
            return eigenvectors, eigenvalues

        eigenvalues = self._as_real_eigenvalues(result)
        return np.sort(eigenvalues)

    def diago_H_near_fermi(self, k_direct_coor, fermi_energy, band_num):
        k_direct_coor = np.asarray(k_direct_coor, dtype=float)
        kpoint_num = k_direct_coor.shape[0]
        eigenvectors = np.zeros([kpoint_num, self.basis_num, band_num], dtype=complex)
        eigenvalues = np.zeros([kpoint_num, band_num], dtype=float)
        for ik, kpoint in enumerate(k_direct_coor):
            Hk_sparse, Sk_sparse = self.get_HSk_sparse(kpoint)
            eigenvectors[ik], eigenvalues[ik] = self._solve_sparse_near_sigma(
                Hk_sparse, Sk_sparse, fermi_energy, band_num, True
            )
        return eigenvectors, eigenvalues

    def diago_H_eigenvaluesOnly_near_fermi(self, k_direct_coor, fermi_energy, band_num):
        k_direct_coor = np.asarray(k_direct_coor, dtype=float)
        kpoint_num = k_direct_coor.shape[0]
        eigenvalues = np.zeros([kpoint_num, band_num], dtype=float)
        for ik, kpoint in enumerate(k_direct_coor):
            Hk_sparse, Sk_sparse = self.get_HSk_sparse(kpoint)
            eigenvalues[ik] = self._solve_sparse_near_sigma(
                Hk_sparse, Sk_sparse, fermi_energy, band_num, False
            )
        return eigenvalues

    def get_total_berry_curvature_fermi(self, k_direct_coor, fermi_energy, mode):
        kpoint_num = k_direct_coor.shape[0]
        total_berry_curvature = np.zeros([kpoint_num, 3], dtype=float)
        self.tb_solver.get_total_berry_curvature_fermi(k_direct_coor, fermi_energy, mode, total_berry_curvature)

        return total_berry_curvature

    def get_total_berry_curvature_occupiedNumber(self, k_direct_coor, occupiedNumber, mode):
        kpoint_num = k_direct_coor.shape[0]
        total_berry_curvature = np.zeros([kpoint_num, 3], dtype=float)
        self.tb_solver.get_total_berry_curvature_occupiedNumber(k_direct_coor, occupiedNumber, mode, total_berry_curvature)

        return total_berry_curvature

    def get_berry_phase_of_loop(self, k_direct_coor_loop, occupiedNumber):
        phase = self.tb_solver.get_berry_phase_of_loop(k_direct_coor_loop, occupiedNumber)

        return phase

    def get_wilson_loop(self, k_direct_coor_loop, occupiedNumber):
        wilson_phase = self.tb_solver.get_wilson_loop(k_direct_coor_loop, occupiedNumber)

        return wilson_phase

    def get_optical_conductivity(self, nspin, omega_num, domega, start_omega, eta, occupiedNumber, k_direct_coor, total_kpoint_num, method=1, use_fermi=False, fermi_energy=0.0):
        optical_conductivity = np.zeros([9, omega_num], dtype=complex)
        dielectric_function = np.zeros([9, omega_num], dtype=complex)
        self.tb_solver.get_optical_conductivity_by_kubo(nspin, omega_num, domega, start_omega, eta, occupiedNumber, use_fermi, fermi_energy, k_direct_coor, total_kpoint_num, method, optical_conductivity, dielectric_function)

        return optical_conductivity, dielectric_function

    def get_shift_current(self, nspin, omega_num, domega, start_omega, smearing_method, eta, occupiedNumber, k_direct_coor, total_kpoint_num, method=1):
        shift_current = np.zeros([18, omega_num], dtype=float)
        self.tb_solver.get_shift_current(nspin, omega_num, domega, start_omega, smearing_method, eta, occupiedNumber, k_direct_coor, total_kpoint_num, method, shift_current)

        return shift_current
    
    def get_shift_current_n_m_pair(self, nspin, omega_num, domega, start_omega, smearing_method, eta, occupiedNumber, k_direct_coor, total_kpoint_num, n_occ, m_unocc, method=1):
        shift_current = np.zeros([18, omega_num], dtype=float)
        self.tb_solver.get_shift_current_n_m_pair(nspin, omega_num, domega, start_omega, smearing_method, eta, occupiedNumber, k_direct_coor, total_kpoint_num, n_occ, m_unocc, method, shift_current)

        return shift_current
    
    def get_second_harmonic(self, method,eta,omega_num, domega, start_omega, fermi_energy, total_kpoint_num,k_direct_coor):
        second_harmonic = np.zeros([27, omega_num], dtype=complex)
        self.tb_solver.get_second_harmonic(method,eta,omega_num, domega, start_omega, fermi_energy, total_kpoint_num,k_direct_coor, second_harmonic)
        return second_harmonic
    
    def get_pockels(self,omega_num, domega, start_omega, fermi_energy, omega1,total_kpoint_num,k_direct_coor):
        pockels = np.zeros([27, omega_num], dtype=complex)
        self.tb_solver.get_pockels(omega_num, domega, start_omega, fermi_energy, omega1,total_kpoint_num,k_direct_coor, pockels)

        return pockels
    def get_bcd(self,omega_num, domega, start_omega,total_kpoint_num,k_direct_coor):
        bcd = np.zeros([9, omega_num], dtype=float)
        self.tb_solver.get_bcd(omega_num, domega, start_omega,total_kpoint_num,k_direct_coor, bcd)

        return bcd
    def get_velocity_matrix(self, k_direct_coor):
        kpoint_num = k_direct_coor.shape[0]
        eigenvalues = np.zeros([kpoint_num, self.basis_num], dtype=float)
        velocity_matrix = np.zeros([kpoint_num, 3, self.basis_num, self.basis_num], dtype=complex)
        self.tb_solver.get_velocity_matrix(k_direct_coor, eigenvalues, velocity_matrix)

        return eigenvalues, velocity_matrix

    def get_bandunfolding(self, M_matrix, kvect_direct, ecut, min_bandindex, max_bandindex, nspin):
        kpoint_num = kvect_direct.shape[0]
        select_band_num = max_bandindex - min_bandindex + 1
        P = np.zeros((kpoint_num, select_band_num), dtype=float)
        E = np.zeros((kpoint_num, select_band_num), dtype=float)
        self.tb_solver.get_bandunfolding(M_matrix, kvect_direct, ecut, min_bandindex, max_bandindex, nspin, P, E)

        return P, E
    
    def get_bandunfolding_spin_texture(self, M_matrix, kvect_direct, ecut, min_bandindex, max_bandindex, nspin):
        kpoint_num = kvect_direct.shape[0]
        select_band_num = max_bandindex - min_bandindex + 1
        P = np.zeros((kpoint_num, select_band_num), dtype=float)
        P_sx = np.zeros((kpoint_num, select_band_num), dtype=float)
        P_sy = np.zeros((kpoint_num, select_band_num), dtype=float)
        P_sz = np.zeros((kpoint_num, select_band_num), dtype=float)
        E = np.zeros((kpoint_num, select_band_num), dtype=float)
        self.tb_solver.get_bandunfolding_spin_texture(M_matrix, kvect_direct, ecut, min_bandindex, max_bandindex, nspin, P, P_sx, P_sy, P_sz, E)

        return P, P_sx, P_sy, P_sz, E

    def get_rnm_drnm_k(self, k_direct_coor):
        kpoint_num = k_direct_coor.shape[0]
        r_nm = np.zeros((kpoint_num, 3, self.basis_num, self.basis_num), dtype=complex)
        dr_nm = np.zeros((kpoint_num, 9, self.basis_num, self.basis_num), dtype=complex)
        self.tb_solver.get_rnm_drnm_k(k_direct_coor, r_nm, dr_nm)

        return r_nm, dr_nm
    
    def get_velocity_basis_k(self, k_direct_coor):
        kpoint_num = k_direct_coor.shape[0]
        velocity_basis_k = np.zeros([kpoint_num, 3, self.basis_num, self.basis_num], dtype=complex)
        self.tb_solver.get_velocity_basis_k(k_direct_coor, velocity_basis_k)

        return velocity_basis_k
    
    def get_inner_product_twoPoints(self, k_direct_coor_start, k_direct_coor_end):
        inner_product = np.zeros([self.basis_num, self.basis_num], dtype=complex)
        k_diret_coor_start_and_end = np.vstack((k_direct_coor_start, k_direct_coor_end))
        self.tb_solver.get_inner_product_twoPoints(k_diret_coor_start_and_end, inner_product)

        return inner_product