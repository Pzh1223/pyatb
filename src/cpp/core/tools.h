#ifndef TOOL_H
#define TOOL_H

#include <vector>
#include <string>
#include "use_eigen.h"
#include "omp.h"

using namespace Eigen;

class tools
{
public:
    static MatrixXcd convert_tril(
        const int &basis_num, 
        const VectorXcd &upperTriangleOfDenseMatrix
    );

    static void diagonalize_SelfAdjointMatrix_eigenvaluesOnly(
        const MatrixXcd &H_k,
        VectorXd &eigenvalues
    );

    static void diagonalize_GeneralizedSelfAdjointMatrix_eigenvaluesOnly_1k(
        const MatrixXcd &H_k, 
        const MatrixXcd &S_k, 
        VectorXd &eigenvalues
    );

    static void diagonalize_GeneralizedSelfAdjointMatrix_eigenvaluesOnly_range_1k(
        const MatrixXcd &H_k, 
        const MatrixXcd &S_k, 
        const int &lower_eigen_index, // counting from 1.
        const int &upper_eigen_index, // counting from 1, upper_band_index >= lower_band_index
        VectorXd &eigenvalues
    );

    static void diagonalize_GeneralizedSelfAdjointMatrix_1k(
        const MatrixXcd &H_k, 
        const MatrixXcd &S_k, 
        MatrixXcd &eigenvectors,
        VectorXd &eigenvalues
    );

    static void diagonalize_GeneralizedSelfAdjointMatrix_range_1k(
        const MatrixXcd &H_k, 
        const MatrixXcd &S_k, 
        const int &lower_eigen_index, // counting from 1.
        const int &upper_eigen_index, // counting from 1, upper_band_index >= lower_band_index
        MatrixXcd &eigenvectors,
        VectorXd &eigenvalues
    );

    static void diagonalize_GeneralizedSelfAdjointMatrix_eigenvaluesOnly(
        const std::vector<MatrixXcd> &H_k, 
        const std::vector<MatrixXcd> &S_k, 
        std::vector<VectorXd> &eigenvalues
    );

    static void diagonalize_GeneralizedSelfAdjointMatrix_eigenvaluesOnly_range(
        const std::vector<MatrixXcd> &H_k, 
        const std::vector<MatrixXcd> &S_k, 
        const int &lower_eigen_index, // counting from 1.
        const int &upper_eigen_index, // counting from 1, upper_band_index >= lower_band_index
        std::vector<VectorXd> &eigenvalues
    );

    static void diagonalize_GeneralizedSelfAdjointMatrix(
        const std::vector<MatrixXcd> &H_k, 
        const std::vector<MatrixXcd> &S_k, 
        std::vector<MatrixXcd> &eigenvectors,
        std::vector<VectorXd> &eigenvalues
    );

    static void diagonalize_GeneralizedSelfAdjointMatrix_range(
        const std::vector<MatrixXcd> &H_k, 
        const std::vector<MatrixXcd> &S_k, 
        const int &lower_eigen_index, // counting from 1.
        const int &upper_eigen_index, // counting from 1, upper_band_index >= lower_band_index
        std::vector<MatrixXcd> &eigenvectors,
        std::vector<VectorXd> &eigenvalues
    );

    // ARPACK shift-invert (mode=3) for generalized Hermitian eigenproblem H*v = lambda*S*v.
    // Finds nev eigenvalues closest to sigma (shift) via reverse-communication Lanczos.
    // Inner linear solve (H - sigma*S)*y = S*x is done with a dense LU factorization.
    // On success, eigenvalues and eigenvectors are sorted by ascending eigenvalue.
    // Returns false if ARPACK failed to converge (caller should fall back to LAPACK).
    static bool diagonalize_arpack_shift_invert_1k(
        const MatrixXcd &H_k,
        const MatrixXcd &S_k,
        const int &nev,           // number of eigenvalues/vectors to compute
        const double &sigma,      // shift (typically Fermi energy in eV)
        const int &ncv,           // Krylov subspace size; must satisfy nev+1 <= ncv <= n
        const double &tol,        // relative convergence tolerance (0.0 = machine precision)
        const int &maxiter,       // maximum number of Arnoldi update iterations
        VectorXd &eigenvalues,    // output: sorted real eigenvalues, length nev
        MatrixXcd &eigenvectors   // output: corresponding eigenvectors, n x nev
    );

    // Eigenvalue-only variant (no eigenvector computation).
    static bool diagonalize_arpack_shift_invert_eigenvaluesOnly_1k(
        const MatrixXcd &H_k,
        const MatrixXcd &S_k,
        const int &nev,
        const double &sigma,
        const int &ncv,
        const double &tol,
        const int &maxiter,
        VectorXd &eigenvalues
    );

    // PARPACK backend hook. The current implementation keeps the MPI communicator
    // explicit at the API boundary so the numerical kernel can be upgraded without
    // changing the Python/C++ call chain again.
    static bool diagonalize_parpack_shift_invert_1k(
        const MatrixXcd &H_k,
        const MatrixXcd &S_k,
        const int &nev,
        const double &sigma,
        const int &ncv,
        const double &tol,
        const int &maxiter,
        const int &mpi_comm_f,
        VectorXd &eigenvalues,
        MatrixXcd &eigenvectors
    );

    static bool diagonalize_parpack_shift_invert_eigenvaluesOnly_1k(
        const MatrixXcd &H_k,
        const MatrixXcd &S_k,
        const int &nev,
        const double &sigma,
        const int &ncv,
        const double &tol,
        const int &maxiter,
        const int &mpi_comm_f,
        VectorXd &eigenvalues
    );
};



#endif