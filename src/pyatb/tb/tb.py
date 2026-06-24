import warnings
import numpy as np
from pyatb.tb.solver import solver
from pyatb.io import abacus_read_stru

# Memory threshold (in bytes) above which dense conversion is skipped and the
# sparse solver is used automatically.  Default: 1 GiB.
_DENSE_MEMORY_THRESHOLD = 1 * 1024 ** 3

class tb:
    def __init__(self, nspin, lattice_constant, lattice_vector, max_kpoint_num=None, **kwarg):
        self.nspin = nspin
        self.lattice_constant = lattice_constant
        self.lattice_vector = lattice_vector
        self.reciprocal_vector = np.linalg.inv(self.lattice_vector).transpose()
        self.unit_cell_volume = np.linalg.det(lattice_vector) * lattice_constant**3
        self.read_atom_position = False
        self.read_atom_orb = False
        
        # The maximum k-point number that can be calculated.
        if max_kpoint_num is None:
            self.max_kpoint_num = 8000
        else:
            self.max_kpoint_num = max_kpoint_num

        if nspin != 2:
            self.tb_solver = solver(self.lattice_constant, self.lattice_vector)
        else:
            self.tb_solver_up = solver(self.lattice_constant, self.lattice_vector)
            self.tb_solver_dn = solver(self.lattice_constant, self.lattice_vector)

    @staticmethod
    def _check_dense_memory(XR_matrix, name='XR'):
        """Return True when converting *XR_matrix* to dense is safe.

        Emits a warning and returns False when the resulting dense array would
        exceed ``_DENSE_MEMORY_THRESHOLD`` bytes (default 1 GiB), so callers
        can automatically fall back to the sparse solver.
        """
        rows, cols = XR_matrix.shape
        # complex128 occupies 16 bytes per element
        required_bytes = rows * cols * 16
        if required_bytes > _DENSE_MEMORY_THRESHOLD:
            warnings.warn(
                f"Dense conversion of {name} would require "
                f"{required_bytes / 1024**3:.2f} GiB "
                f"(shape {XR_matrix.shape}, dtype complex128), which exceeds "
                f"the {_DENSE_MEMORY_THRESHOLD / 1024**3:.0f} GiB threshold. "
                "Automatically switching to the sparse solver. "
                "Set isSparse=True explicitly to suppress this warning.",
                ResourceWarning,
                stacklevel=3,
            )
            return False
        return True

    def set_solver_HSR(self, HR, SR, isSparse=False):
        # Check whether HR and SR are consistent
        if HR.des != 'H':
            raise ValueError('HR parameter error !!')
        if SR.des != 'S':
            raise ValueError('SR parameter error !!')
        if not (HR.R_direct_coor == SR.R_direct_coor).all():
            raise ValueError('HR and SR mismatch !!')
        if HR.basis_num != SR.basis_num:
            raise ValueError('HR and SR mismatch !!')

        if not isSparse and not self._check_dense_memory(HR.XR, 'HR'):
            isSparse = True

        self.HSR_iSsparse = isSparse

        if isSparse:
            self.tb_solver.set_HSR_sparse(HR.R_num, HR.R_direct_coor, HR.basis_num, HR.XR, SR.XR)
        else:
            temp_HR = HR.XR.toarray()
            temp_SR = SR.XR.toarray()
            self.tb_solver.set_HSR(HR.R_num, HR.R_direct_coor, HR.basis_num, temp_HR, temp_SR)

        self.R_num = HR.R_num
        self.R_direct_coor = HR.R_direct_coor
        self.basis_num = HR.basis_num

    def set_solver_HSR_spin2(self, HR_up, HR_dn, SR, isSparse=False):
        if self.nspin != 2:
            raise ValueError('nspin is not equal to 2, the function cannot be called')

        # Check whether HR and SR are consistent
        if HR_up.des != 'H':
            raise ValueError('HR parameter error !!')
        if HR_dn.des != 'H':
            raise ValueError('HR parameter error !!')
        if SR.des != 'S':
            raise ValueError('SR parameter error !!')
        if not (HR_up.R_direct_coor == SR.R_direct_coor).all():
            raise ValueError('HR and SR mismatch !!')
        if not (HR_dn.R_direct_coor == SR.R_direct_coor).all():
            raise ValueError('HR and SR mismatch !!')
        if HR_up.basis_num != SR.basis_num or HR_dn.basis_num != SR.basis_num:
            raise ValueError('HR and SR mismatch !!')

        if not isSparse and (
            not self._check_dense_memory(HR_up.XR, 'HR_up')
            or not self._check_dense_memory(HR_dn.XR, 'HR_dn')
        ):
            isSparse = True

        self.HSR_iSsparse = isSparse

        if isSparse:
            self.tb_solver_up.set_HSR_sparse(HR_up.R_num, HR_up.R_direct_coor, HR_up.basis_num, HR_up.XR, SR.XR)
            self.tb_solver_dn.set_HSR_sparse(HR_dn.R_num, HR_dn.R_direct_coor, HR_dn.basis_num, HR_dn.XR, SR.XR)
        else:
            temp_HR_up = HR_up.XR.toarray()
            temp_HR_dn = HR_dn.XR.toarray()
            temp_SR = SR.XR.toarray()
            self.tb_solver_up.set_HSR(HR_up.R_num, HR_up.R_direct_coor, HR_up.basis_num, temp_HR_up, temp_SR)
            self.tb_solver_dn.set_HSR(HR_dn.R_num, HR_dn.R_direct_coor, HR_dn.basis_num, temp_HR_dn, temp_SR)

        self.R_num = HR_up.R_num
        self.R_direct_coor = HR_up.R_direct_coor
        self.basis_num = HR_up.basis_num

    def set_solver_rR(self, rR_x, rR_y, rR_z, isSparse=False):
        try:
            self.HSR_iSsparse
        except AttributeError:
            raise RuntimeError(
                'set_solver_rR() must be executed after set_solver_HSR() or set_solver_HSR_spin2()'
            )

        # If the HSR solver was automatically promoted to sparse (e.g. due to
        # a memory-threshold auto-switch), silently follow the same setting so
        # the consistency check below does not raise a false error.
        if not isSparse and self.HSR_iSsparse:
            isSparse = True
        elif not isSparse and (
            not self._check_dense_memory(rR_x.XR, 'rR_x')
            or not self._check_dense_memory(rR_y.XR, 'rR_y')
            or not self._check_dense_memory(rR_z.XR, 'rR_z')
        ):
            isSparse = True

        if self.HSR_iSsparse != isSparse:
            raise ValueError('isSparse must be consistent for rR and HSR')

        # Check whether HR and rR are consistent
        if rR_x.des != 'r_x' or rR_y.des != 'r_y' or rR_z.des != 'r_z':
            raise ValueError('rR parameter error !!')
        if not (rR_x.R_direct_coor == self.R_direct_coor).all():
            raise ValueError('HR and rR_x mismatch !!')
        if not (rR_y.R_direct_coor == self.R_direct_coor).all():
            raise ValueError('HR and rR_y mismatch !!')
        if not (rR_z.R_direct_coor == self.R_direct_coor).all():
            raise ValueError('HR and rR_z mismatch !!')
        if rR_x.basis_num != self.basis_num or rR_y.basis_num != self.basis_num or rR_z.basis_num != self.basis_num:
            raise ValueError('HR and rR mismatch !!')

        if isSparse:
            if self.nspin != 2:
                self.tb_solver.set_rR_sparse(rR_x.XR, rR_y.XR, rR_z.XR)
            else:
                self.tb_solver_up.set_rR_sparse(rR_x.XR, rR_y.XR, rR_z.XR)
                self.tb_solver_dn.set_rR_sparse(rR_x.XR, rR_y.XR, rR_z.XR)
        else:
            temp_rR_x = rR_x.XR.toarray()
            temp_rR_y = rR_y.XR.toarray()
            temp_rR_z = rR_z.XR.toarray()
            if self.nspin != 2:
                self.tb_solver.set_rR(temp_rR_x, temp_rR_y, temp_rR_z)
            else:
                self.tb_solver_up.set_rR(temp_rR_x, temp_rR_y, temp_rR_z)
                self.tb_solver_dn.set_rR(temp_rR_x, temp_rR_y, temp_rR_z)

    def direct_to_cartesian_kspace(self, k_direct_coor):
        """

        Convert fractional coordinates of k points to Cartesian coordinates.

        Args:
            k_direct_coor: float-ndarray, shape is (any size, 3), set of k points, row vector.
        
        """
        kvect_cartesian_coor = k_direct_coor @ self.reciprocal_vector * 2 * np.pi / self.lattice_constant
        return kvect_cartesian_coor

    def cartesian_to_direct_kspace(self, kvect_cartesian_coor):
        """

        Convert Cartesian coordinates of k points to fractional coordinates.

        Args:
            kvect_cartesian_coor: float-ndarray, shape is (any size, 3), set of k points, row vector.
        
        """
        k_direct_coor = kvect_cartesian_coor @ np.linalg.inv(self.reciprocal_vector) * self.lattice_constant / 2 / np.pi
        return k_direct_coor

    def read_stru(self, stru_file, need_orb=False):

        if not self.read_atom_position:
            self.stru_atom = abacus_read_stru.read_stru(stru_file)
            self.atom_type_num = len(self.stru_atom)
            self.total_atom_num = 0

            for i in self.stru_atom:
                atom_type = i.species
                na = i.atom_num
                i.cartesian_coor = i.cartesian_coor / self.lattice_constant
                tau_car = i.cartesian_coor
                self.total_atom_num += na

                if self.nspin != 2:
                    self.tb_solver.set_single_atom_position(atom_type, na, tau_car)
                else:
                    self.tb_solver_up.set_single_atom_position(atom_type, na, tau_car)
                    self.tb_solver_dn.set_single_atom_position(atom_type, na, tau_car)

            self.read_atom_position = True

        
        if not self.read_atom_orb and need_orb:
            for i in self.stru_atom:
                i.read_numerical_orb()

            for i in range(len(self.stru_atom)):
                nwl = self.stru_atom[i].l_max
                l_nchi = self.stru_atom[i].orbital_num
                mesh = self.stru_atom[i].mesh
                dr = self.stru_atom[i].dr
                numerical_orb = self.stru_atom[i].orbit

                if self.nspin != 2:
                    self.tb_solver.set_single_atom_orb(i, nwl, l_nchi, mesh, dr, numerical_orb)
                else:
                    self.tb_solver_up.set_single_atom_orb(i, nwl, l_nchi, mesh, dr, numerical_orb)
                    self.tb_solver_dn.set_single_atom_orb(i, nwl, l_nchi, mesh, dr, numerical_orb)
            
            self.read_atom_orb = True

        return None

