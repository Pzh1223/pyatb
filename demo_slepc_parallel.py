import sys
try:
    from mpi4py import MPI
    from petsc4py import PETSc
    from slepc4py import SLEPc
except ImportError as e:
    print(f"ImportError: {e}. petsc4py and slepc4py are required for this demo.")
    sys.exit(0)

def build_distributed_matrices(N, comm):
    # This simulates distributing the Hamiltonian and Overlap matrix across MPI nodes
    H = PETSc.Mat().create(comm=comm)
    H.setSizes([N, N])
    H.setType('mpiAIJ') # Sparse matrix
    H.setUp()
    
    S = PETSc.Mat().create(comm=comm)
    S.setSizes([N, N])
    S.setType('mpiAIJ')
    S.setUp()

    rstart, rend = H.getOwnershipRange()
    
    # Fill local rows
    for i in range(rstart, rend):
        # Dummy Hamiltonian: Diagonal with some off-diagonals
        H.setValue(i, i, 2.0 * i / N)
        if i > 0:
            H.setValue(i, i-1, -0.5)
        if i < N - 1:
            H.setValue(i, i+1, -0.5)
        
        # Dummy Overlap: Identity
        S.setValue(i, i, 1.0)
        
    H.assemblyBegin()
    H.assemblyEnd()
    S.assemblyBegin()
    S.assemblyEnd()
    
    return H, S

def solve_eigenproblem(H, S, nev, target, comm):
    eps = SLEPc.EPS().create(comm=comm)
    eps.setOperators(H, S)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    
    eps.setTarget(target)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setDimensions(nev=nev)
    
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    
    eps.solve()
    
    nconv = eps.getConverged()
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"Converged eigenpairs: {nconv}")
        
    eigenvalues = []
    if nconv > 0:
        for i in range(min(nconv, nev)):
            val = eps.getEigenvalue(i)
            eigenvalues.append(val.real)
            if rank == 0:
                print(f"Eigenvalue {i}: {val.real:.6f}")
                
    return eigenvalues

def main():
    # Setup MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    PETSc.Sys.Print(f"Starting PETSc/SLEPc demo with {size} processes.")
    
    N = 1000 # Matrix dimension
    nev = 5  # Number of eigenvalues to compute
    target = 0.5 # Target eigenvalue (like fermi_energy)
    
    H, S = build_distributed_matrices(N, comm)
    
    PETSc.Sys.Print("Matrices assembled. Starting solver...")
    
    eigenvalues = solve_eigenproblem(H, S, nev, target, comm)
    
    if rank == 0:
        print("Demo completed successfully!")

if __name__ == "__main__":
    main()
