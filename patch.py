with open("src/pyatb/fermi/band_structure.py", "r") as f:
    lines = f.readlines()

import_petsc = """try:
    from petsc4py import PETSc
    from slepc4py import SLEPc
    HAS_SLEPC = True
except ImportError:
    HAS_SLEPC = False
"""

for i, line in enumerate(lines):
    if "from scipy.sparse import csr_matrix" in line:
        lines.insert(i + 1, import_petsc)
        break

solver_logic = """                    if getattr(self, 'solver', 'dense') in ['arpack', 'sparse', 'parpack']:
                        Hk_list = self.__tb_solver[ispin].get_Hk(ik_process.k_direct_coor_local)
                        Sk_list = self.__tb_solver[ispin].get_Sk(ik_process.k_direct_coor_local)
                        
                        eigenvalues_list = []
                        if self.wf_collect:
                            eigenvectors_list = []
                        
                        for i_k in range(kpoint_num):
                            Hk = Hk_list[i_k]
                            Sk = Sk_list[i_k]
                            # Lanczos solvers cannot compute all eigenvalues. They require k < N.
                            # We set the limit to N-1 for safely using the sparse solver.
                            if cal_band_num >= basis_num - 1:
                                if self.wf_collect:
                                    val, vec = eigh(Hk, b=Sk)
                                    eigenvalues_list.append(val)
                                    eigenvectors_list.append(vec)
                                else:
                                    val = eigvalsh(Hk, b=Sk)
                                    eigenvalues_list.append(val)
                            elif getattr(self, 'solver', 'dense') == 'parpack':
                                if not HAS_SLEPC:
                                    raise ImportError("petsc4py and slepc4py are required to use solver='parpack'. Please install them or use solver='arpack'.")
                                Hk_sparse = csr_matrix(Hk)
                                Sk_sparse = csr_matrix(Sk)
                                
                                H_petsc = PETSc.Mat().createAIJ(size=Hk_sparse.shape, csr=(Hk_sparse.indptr, Hk_sparse.indices, Hk_sparse.data), comm=PETSc.COMM_SELF)
                                S_petsc = PETSc.Mat().createAIJ(size=Sk_sparse.shape, csr=(Sk_sparse.indptr, Sk_sparse.indices, Sk_sparse.data), comm=PETSc.COMM_SELF)
                                
                                eps = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
                                eps.setOperators(H_petsc, S_petsc)
                                eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
                                eps.setDimensions(nev=cal_band_num)
                                eps.setTarget(self.fermi_energy)
                                eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
                                
                                st = eps.getST()
                                st.setType(SLEPc.ST.Type.SINVERT)
                                eps.setType(SLEPc.EPS.Type.ARPACK)
                                
                                eps.solve()
                                
                                nconv = eps.getConverged()
                                if nconv < cal_band_num:
                                    print(f"Warning: SLEPc PARPACK only converged {nconv} eigenvalues out of requested {cal_band_num}")
                                
                                vals = []
                                vecs = []
                                if self.wf_collect:
                                    vr, vi = H_petsc.createVecs()
                                    
                                for idx_eig in range(min(nconv, cal_band_num)):
                                    v = eps.getEigenvalue(idx_eig)
                                    vals.append(v.real)
                                    if self.wf_collect:
                                        eps.getEigenvector(idx_eig, vr, vi)
                                        # Handle real and complex PETSc builds
                                        if np.iscomplexobj(vr.getArray()):
                                            vecs.append(vr.getArray().copy())
                                        else:
                                            vecs.append(vr.getArray() + 1j * vi.getArray())
                                            
                                val = np.array(vals)
                                if self.wf_collect:
                                    vec = np.array(vecs).T
                                    idx = np.argsort(val)
                                    val = val[idx]
                                    vec = vec[:, idx]
                                    eigenvalues_list.append(val)
                                    eigenvectors_list.append(vec)
                                else:
                                    idx = np.argsort(val)
                                    val = val[idx]
                                    eigenvalues_list.append(val)
                            else:
                                Hk_sparse = csr_matrix(Hk)
                                Sk_sparse = csr_matrix(Sk)
                                val, vec = eigsh(Hk_sparse, k=cal_band_num, M=Sk_sparse, sigma=self.fermi_energy, which='LM')
                                idx = np.argsort(val)
                                val = val[idx]
                                vec = vec[:, idx]
                                eigenvalues_list.append(val)
                                if self.wf_collect:
                                    eigenvectors_list.append(vec)"""

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if "if getattr(self, 'solver', 'dense') in ['arpack', 'sparse']:" in line:
        start_idx = i
    elif "eigenvalues = np.array(eigenvalues_list, dtype=float)" in line and start_idx != -1:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    lines[start_idx:end_idx] = [solver_logic + "\n"]

with open("src/pyatb/fermi/band_structure.py", "w") as f:
    f.writelines(lines)
