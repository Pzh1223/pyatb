#include "tools.h"
#include <iostream>
#include <algorithm>
#include <cstring>

MatrixXcd tools::convert_tril(
    const int &basis_num, 
    const VectorXcd &upperTriangleOfDenseMatrix
)
{
    MatrixXcd container;
    container.setZero(basis_num, basis_num);

    int count = 0;
    for (int row = 0; row < basis_num; row++)
    {
        for (int col = row; col < basis_num; col++)
        {
            container(row, col) = upperTriangleOfDenseMatrix(count);
            count++;
        }
    }

    return container;
}

void tools::diagonalize_SelfAdjointMatrix_eigenvaluesOnly(
    const MatrixXcd &H_k,
    VectorXd &eigenvalues
)
{
    SelfAdjointEigenSolver<MatrixXcd> eigenSolver;
    eigenSolver.compute(H_k, EigenvaluesOnly);
    eigenvalues = eigenSolver.eigenvalues();
}

void tools::diagonalize_GeneralizedSelfAdjointMatrix_eigenvaluesOnly_1k(
    const MatrixXcd &H_k, 
    const MatrixXcd &S_k, 
    VectorXd &eigenvalues
)
{
    GeneralizedSelfAdjointEigenSolver<MatrixXcd> eigenSolver;
    eigenSolver.compute(H_k, S_k, EigenvaluesOnly);
    eigenvalues = eigenSolver.eigenvalues();
}

void tools::diagonalize_GeneralizedSelfAdjointMatrix_eigenvaluesOnly_range_1k(
    const MatrixXcd &H_k, 
    const MatrixXcd &S_k, 
    const int &lower_eigen_index, // counting from 1.
    const int &upper_eigen_index, // counting from 1, upper_band_index >= lower_band_index
    VectorXd &eigenvalues
)
{
    MatrixXcd copy_H_k = H_k;
    MatrixXcd copy_S_k = S_k;

    int itype = 1; 
    char jobz = 'N'; 
    char range = 'I'; 
    double vl = 0.0;
    double vu = 0.0;
    lapack_int il = lower_eigen_index;
    lapack_int iu = upper_eigen_index;
    double abstol = 0.0;
    lapack_int n = H_k.rows();
    lapack_int lda = n;
    lapack_int ldb = n;

    std::vector<std::complex<double>> z(n * n);
    std::vector<double> w(n);
    std::vector<lapack_int> ifail(n);

    lapack_int m;
    lapack_int info;

    info = LAPACKE_zhegvx(
        LAPACK_COL_MAJOR,
        itype,
        jobz,
        range,
        'U',
        n,
        reinterpret_cast<lapack_complex_double*>(copy_H_k.data()), lda,
        reinterpret_cast<lapack_complex_double*>(copy_S_k.data()), ldb,
        vl, vu, il, iu,
        abstol,
        &m,
        w.data(),
        reinterpret_cast<lapack_complex_double*>(z.data()), n,
        &ifail[0]
    );

    if (info > 0) {
        std::cerr << "tools.cpp: The algorithm failed to compute eigenvalues." << std::endl;
    } else if (info < 0) {
        std::cerr << "tools.cpp: Argument " << -info << " had an illegal value." << std::endl;
    }

    eigenvalues = VectorXd::Zero(m);
    for(int i = 0; i < m; ++i)
    {
        eigenvalues[i] = w[i];
    }

}

void tools::diagonalize_GeneralizedSelfAdjointMatrix_1k(
    const MatrixXcd &H_k, 
    const MatrixXcd &S_k, 
    MatrixXcd &eigenvectors,
    VectorXd &eigenvalues
)
{
    GeneralizedSelfAdjointEigenSolver<MatrixXcd> eigenSolver;  
    eigenSolver.compute(H_k, S_k);
    eigenvectors = eigenSolver.eigenvectors();
    eigenvalues = eigenSolver.eigenvalues();
}

void tools::diagonalize_GeneralizedSelfAdjointMatrix_range_1k(
    const MatrixXcd &H_k, 
    const MatrixXcd &S_k, 
    const int &lower_eigen_index, // counting from 1.
    const int &upper_eigen_index, // counting from 1, upper_band_index >= lower_band_index
    MatrixXcd &eigenvectors,
    VectorXd &eigenvalues
)
{
    MatrixXcd copy_H_k = H_k;
    MatrixXcd copy_S_k = S_k;

    int itype = 1; 
    char jobz = 'V'; 
    char range = 'I'; 
    double vl = 0.0;
    double vu = 0.0;
    lapack_int il = lower_eigen_index;
    lapack_int iu = upper_eigen_index;
    double abstol = 0.0;
    lapack_int n = H_k.rows();
    lapack_int lda = n;
    lapack_int ldb = n;

    std::vector<std::complex<double>> z(n * n);
    std::vector<double> w(n);
    std::vector<lapack_int> ifail(n);

    lapack_int m;
    lapack_int info;

    info = LAPACKE_zhegvx(
        LAPACK_COL_MAJOR,
        itype,
        jobz,
        range,
        'U',
        n,
        reinterpret_cast<lapack_complex_double*>(copy_H_k.data()), lda,
        reinterpret_cast<lapack_complex_double*>(copy_S_k.data()), ldb,
        vl, vu, il, iu,
        abstol,
        &m,
        w.data(),
        reinterpret_cast<lapack_complex_double*>(z.data()), n,
        &ifail[0]
    );

    if (info > 0) {
        std::cerr << "tools.cpp: The algorithm failed to compute eigenvalues." << std::endl;
    } else if (info < 0) {
        std::cerr << "tools.cpp: Argument " << -info << " had an illegal value." << std::endl;
    }

    eigenvalues = VectorXd::Zero(m);
    for(int i = 0; i < m; ++i)
    {
        eigenvalues[i] = w[i];
    }

    eigenvectors = MatrixXcd::Zero(n, m);
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            eigenvectors(j, i) = z[i * n + j];
        }
    }

    // Check for unconverged eigenvectors
    bool has_ifail = false;
    for(int i = 0; i < m; ++i)
    {
        if(ifail[i] > 0)
        {
            if(!has_ifail)
            {
                std::cout << "tools.cpp: The following eigenvectors failed to converge:" << std::endl;
                has_ifail = true;
            }
            std::cout << "Eigenvector " << i+1 << " failed to converge." << std::endl;
        }
    }

}

void tools::diagonalize_GeneralizedSelfAdjointMatrix_eigenvaluesOnly(
    const std::vector<MatrixXcd> &H_k, 
    const std::vector<MatrixXcd> &S_k, 
    std::vector<VectorXd> &eigenvalues
)
{
    int kpoint_num = H_k.size();
    int max_num_threads = omp_get_max_threads();

    eigenvalues.resize(kpoint_num);

    if (max_num_threads > kpoint_num)
    {
        GeneralizedSelfAdjointEigenSolver<MatrixXcd> eigenSolver;
        for (int ik = 0; ik < kpoint_num; ik++)
        {
            eigenSolver.compute(H_k[ik], S_k[ik], EigenvaluesOnly);
            eigenvalues[ik] = eigenSolver.eigenvalues();
        }
    }
    else
    {
        GeneralizedSelfAdjointEigenSolver<MatrixXcd> eigenSolver;

        #pragma omp parallel for private(eigenSolver) schedule(static)
        for (int ik = 0; ik < kpoint_num; ik++)
        {
            
            eigenSolver.compute(H_k[ik], S_k[ik], EigenvaluesOnly);
            eigenvalues[ik] = eigenSolver.eigenvalues();
        }
    }
}

void tools::diagonalize_GeneralizedSelfAdjointMatrix_eigenvaluesOnly_range(
    const std::vector<MatrixXcd> &H_k, 
    const std::vector<MatrixXcd> &S_k, 
    const int &lower_eigen_index, // counting from 1.
    const int &upper_eigen_index, // counting from 1, upper_band_index >= lower_band_index
    std::vector<VectorXd> &eigenvalues
)
{
    int kpoint_num = H_k.size();
    int max_num_threads = omp_get_max_threads();

    eigenvalues.resize(kpoint_num);

    if (max_num_threads > kpoint_num)
    {
        for (int ik = 0; ik < kpoint_num; ik++)
        {
            diagonalize_GeneralizedSelfAdjointMatrix_eigenvaluesOnly_range_1k(H_k[ik], S_k[ik], lower_eigen_index, upper_eigen_index, eigenvalues[ik]);
        }
    }
    else
    {
        #pragma omp parallel for schedule(static)
        for (int ik = 0; ik < kpoint_num; ik++)
        {
            diagonalize_GeneralizedSelfAdjointMatrix_eigenvaluesOnly_range_1k(H_k[ik], S_k[ik], lower_eigen_index, upper_eigen_index, eigenvalues[ik]);
        }
    }
    
}

void tools::diagonalize_GeneralizedSelfAdjointMatrix(
    const std::vector<MatrixXcd> &H_k, 
    const std::vector<MatrixXcd> &S_k, 
    std::vector<MatrixXcd> &eigenvectors,
    std::vector<VectorXd> &eigenvalues
)
{
    int kpoint_num = H_k.size();
    int max_num_threads = omp_get_max_threads();

    eigenvectors.resize(kpoint_num);
    eigenvalues.resize(kpoint_num);

    if (max_num_threads > kpoint_num)
    {
        GeneralizedSelfAdjointEigenSolver<MatrixXcd> eigenSolver;
        for (int ik = 0; ik < kpoint_num; ik++)
        {
            eigenSolver.compute(H_k[ik], S_k[ik]);
            eigenvectors[ik] = eigenSolver.eigenvectors();
            eigenvalues[ik] = eigenSolver.eigenvalues();
        }
    }
    else
    {
        GeneralizedSelfAdjointEigenSolver<MatrixXcd> eigenSolver;

        #pragma omp parallel for private(eigenSolver) schedule(static)
        for (int ik = 0; ik < kpoint_num; ik++)
        {
            
            eigenSolver.compute(H_k[ik], S_k[ik]);
            eigenvectors[ik] = eigenSolver.eigenvectors();
            eigenvalues[ik] = eigenSolver.eigenvalues();
        }
    }
}

void tools::diagonalize_GeneralizedSelfAdjointMatrix_range(
    const std::vector<MatrixXcd> &H_k, 
    const std::vector<MatrixXcd> &S_k, 
    const int &lower_eigen_index, // counting from 1.
    const int &upper_eigen_index, // counting from 1, upper_band_index >= lower_band_index
    std::vector<MatrixXcd> &eigenvectors,
    std::vector<VectorXd> &eigenvalues
)
{
    int kpoint_num = H_k.size();
    int max_num_threads = omp_get_max_threads();

    eigenvectors.resize(kpoint_num);
    eigenvalues.resize(kpoint_num);

    if (max_num_threads > kpoint_num)
    {
        for (int ik = 0; ik < kpoint_num; ik++)
        {
            diagonalize_GeneralizedSelfAdjointMatrix_range_1k(H_k[ik], S_k[ik], lower_eigen_index, upper_eigen_index, eigenvectors[ik], eigenvalues[ik]);
        }
    }
    else
    {
        #pragma omp parallel for schedule(static)
        for (int ik = 0; ik < kpoint_num; ik++)
        {
            diagonalize_GeneralizedSelfAdjointMatrix_range_1k(H_k[ik], S_k[ik], lower_eigen_index, upper_eigen_index, eigenvectors[ik], eigenvalues[ik]);
        }
    }

}

// ---------------------------------------------------------------------------
// ARPACK reverse-communication Fortran interfaces
// Naming convention: lowercase + trailing underscore (GNU/Linux gfortran ABI).
// ---------------------------------------------------------------------------
extern "C"
{
    // Standard/Generalized complex non-symmetric Arnoldi update
    void znaupd_(int *ido, const char *bmat, int *n, const char *which,
                 int *nev, double *tol, std::complex<double> *resid,
                 int *ncv, std::complex<double> *v, int *ldv,
                 int *iparam, int *ipntr, std::complex<double> *workd,
                 std::complex<double> *workl, int *lworkl,
                 double *rwork, int *info);

    // Extract Ritz pairs after znaupd convergence
    void zneupd_(int *rvec, const char *howmny, int *select,
                 std::complex<double> *d, std::complex<double> *z, int *ldz,
                 std::complex<double> *sigma, std::complex<double> *workev,
                 const char *bmat, int *n, const char *which, int *nev,
                 double *tol, std::complex<double> *resid, int *ncv,
                 std::complex<double> *v, int *ldv, int *iparam, int *ipntr,
                 std::complex<double> *workd, std::complex<double> *workl,
                 int *lworkl, double *rwork, int *info);
}

// ---------------------------------------------------------------------------
// diagonalize_arpack_shift_invert_1k
//
// Solves the generalized Hermitian eigenproblem  H*v = lambda*S*v  for the
// nev eigenvalues closest to sigma using ARPACK mode=3 (shift-invert).
//
// Operator:  OP x = (H - sigma*S)^{-1} S x
//   => the Ritz values of OP are  mu_i = 1/(lambda_i - sigma)
//   => the nev eigenvalues with |mu_i| largest correspond to the nev
//      eigenvalues of the original problem closest to sigma.
//
// Inner linear solve strategy (Stage B.1 from the development plan):
//   Pre-compute one dense LU factorization of (H - sigma*S), then reuse it
//   for every reverse-communication apply step.  This is O(n^3) once and
//   O(n^2) per ARPACK iteration -- the outer Lanczos loop converges in O(nev)
//   iterations rather than always doing the full n-dimensional reduction.
// ---------------------------------------------------------------------------
bool tools::diagonalize_arpack_shift_invert_1k(
    const MatrixXcd &H_k,
    const MatrixXcd &S_k,
    const int &nev,
    const double &sigma,
    const int &ncv_in,
    const double &tol_in,
    const int &maxiter,
    VectorXd &eigenvalues,
    MatrixXcd &eigenvectors
)
{
    int n = static_cast<int>(H_k.rows());

    // Enforce ARPACK constraint: nev+1 <= ncv <= n
    int ncv = std::max(ncv_in, nev + 2);
    ncv = std::min(ncv, n);

    if (nev <= 0 || nev >= n) {
        std::cerr << "tools: ARPACK nev must satisfy 0 < nev < n (n=" << n << ", nev=" << nev << ").\n";
        return false;
    }
    if (ncv < nev + 1) {
        std::cerr << "tools: ARPACK ncv (" << ncv << ") too small for nev=" << nev << ".\n";
        return false;
    }

    // ---------------------------------------------------------------------------
    // Pre-factorize (H - sigma*S) with dense LU (Eigen partial-pivot LU).
    // ---------------------------------------------------------------------------
    std::complex<double> sigma_c(sigma, 0.0);
    MatrixXcd HsS = H_k - sigma_c * S_k;
    Eigen::PartialPivLU<MatrixXcd> lu(HsS);

    // ---------------------------------------------------------------------------
    // ARPACK setup
    // ---------------------------------------------------------------------------
    const char bmat = 'G';    // generalized problem
    const char *which = "LM"; // largest magnitude of OP = closest to sigma

    double tol = tol_in;
    int ido = 0;
    int ldv = n;
    int lworkl = 3 * ncv * ncv + 5 * ncv;
    int info = 0;

    std::vector<std::complex<double>> resid(n, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> V(n * ncv, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> workd(3 * n, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> workl(lworkl, std::complex<double>(0.0, 0.0));
    std::vector<double> rwork(ncv, 0.0);
    std::vector<int> iparam(11, 0);
    std::vector<int> ipntr(14, 0);

    iparam[0] = 1;       // ISHIFT: use exact shifts
    iparam[2] = maxiter; // MAXITR
    iparam[6] = 3;       // MODE: 3 = shift-invert generalized

    // ---------------------------------------------------------------------------
    // Reverse-communication loop
    // ---------------------------------------------------------------------------
    while (true)
    {
        znaupd_(&ido, &bmat, &n, which,
                &nev, &tol, resid.data(),
                &ncv, V.data(), &ldv,
                iparam.data(), ipntr.data(), workd.data(),
                workl.data(), &lworkl, rwork.data(), &info);

        if (ido == -1 || ido == 1)
        {
            // Apply OP x = (H - sigma*S)^{-1} S x
            // x = workd[ ipntr[0]-1 .. ]   (1-based Fortran index)
            // y = workd[ ipntr[1]-1 .. ]
            // When ido==1, S*x is already stored at workd[ ipntr[2]-1 .. ]
            Eigen::Map<VectorXcd> x(workd.data() + ipntr[0] - 1, n);
            Eigen::Map<VectorXcd> y(workd.data() + ipntr[1] - 1, n);

            if (ido == -1)
            {
                // First call: compute S*x then solve (H-σS)*y = S*x
                VectorXcd Bx = S_k * x;
                y = lu.solve(Bx);
            }
            else
            {
                // ido == 1: B*x already stored at ipntr[2]
                Eigen::Map<const VectorXcd> Bx(workd.data() + ipntr[2] - 1, n);
                y = lu.solve(VectorXcd(Bx));
            }
        }
        else if (ido == 2)
        {
            // Apply B x = S*x
            Eigen::Map<VectorXcd> x(workd.data() + ipntr[0] - 1, n);
            Eigen::Map<VectorXcd> y(workd.data() + ipntr[1] - 1, n);
            y = S_k * x;
        }
        else
        {
            // ido == 99: done (converged or fatal error)
            break;
        }
    }

    if (info < 0)
    {
        std::cerr << "tools: znaupd returned error info=" << info << ".\n";
        return false;
    }
    if (info == 1)
    {
        std::cerr << "tools: znaupd warning: maximum iterations (" << maxiter << ") reached. "
                  << iparam[4] << " of " << nev << " Ritz values converged.\n";
        // Continue with partially converged results.
    }

    // ---------------------------------------------------------------------------
    // Extract Ritz pairs via zneupd
    // ---------------------------------------------------------------------------
    int rvec = 1;       // compute Ritz vectors
    char howmny = 'A';
    std::vector<int> select(ncv, 0);
    std::vector<std::complex<double>> d(nev + 1, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> Z(n * nev, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> workev(2 * ncv, std::complex<double>(0.0, 0.0));
    int info2 = 0;

    zneupd_(&rvec, &howmny, select.data(),
            d.data(), Z.data(), &n,
            &sigma_c, workev.data(),
            &bmat, &n, which, &nev,
            &tol, resid.data(), &ncv,
            V.data(), &ldv, iparam.data(), ipntr.data(),
            workd.data(), workl.data(), &lworkl, rwork.data(), &info2);

    if (info2 != 0)
    {
        std::cerr << "tools: zneupd returned error info=" << info2 << ".\n";
        return false;
    }

    // ---------------------------------------------------------------------------
    // Sort by ascending real part of eigenvalue.
    // For a Hermitian problem the imaginary parts should be negligible.
    // ---------------------------------------------------------------------------
    int nconv = iparam[4];
    int nout  = std::min(nconv, nev);

    std::vector<std::pair<double, int>> sorted;
    sorted.reserve(nout);
    for (int i = 0; i < nout; ++i)
    {
        double imag_threshold = 1e-6 * std::abs(d[i].real()) + 1e-10;
        if (std::abs(d[i].imag()) > imag_threshold)
        {
            std::cerr << "tools: ARPACK eigenvalue " << i << " has non-negligible imaginary part "
                      << d[i].imag() << " (Hermitian problem requires real eigenvalues).\n";
            return false;
        }
        sorted.push_back({d[i].real(), i});
    }
    std::sort(sorted.begin(), sorted.end());

    eigenvalues.resize(nout);
    eigenvectors.resize(n, nout);
    for (int i = 0; i < nout; ++i)
    {
        int idx = sorted[i].second;
        eigenvalues[i] = sorted[i].first;
        for (int j = 0; j < n; ++j)
            eigenvectors(j, i) = Z[static_cast<size_t>(idx) * n + j];
    }

    return true;
}

bool tools::diagonalize_arpack_shift_invert_eigenvaluesOnly_1k(
    const MatrixXcd &H_k,
    const MatrixXcd &S_k,
    const int &nev,
    const double &sigma,
    const int &ncv_in,
    const double &tol_in,
    const int &maxiter,
    VectorXd &eigenvalues
)
{
    MatrixXcd eigenvectors;
    return diagonalize_arpack_shift_invert_1k(
        H_k, S_k, nev, sigma, ncv_in, tol_in, maxiter,
        eigenvalues, eigenvectors);
}