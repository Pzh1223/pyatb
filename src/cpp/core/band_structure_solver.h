#ifndef BAND_STRUCTURE_SOLVER_H
#define BAND_STRUCTURE_SOLVER_H


#include "base_data.h"
#include "xr_operation.h"
#include "tools.h"

class band_structure_solver
{
public:
    static void get_eigenvalues_1k(
        base_data &Base_Data, 
        const VectorXcd &exp_ikR, 
        VectorXd &eigenvalues
    );

    static void get_eigenvalues_range_1k(
        base_data &Base_Data, 
        const VectorXcd &exp_ikR, 
        const int &lower_band_index, // counting from 1.
        const int &upper_band_index, // counting from 1, upper_band_index >= lower_band_index
        VectorXd &eigenvalues
    );
    
    static void get_eigenvalues_eigenvectors_1k(
        base_data &Base_Data, 
        const VectorXcd &exp_ikR, 
        VectorXd &eigenvalues, 
        MatrixXcd &eigenvectors
    );

    static void get_eigenvalues_eigenvectors_range_1k(
        base_data &Base_Data, 
        const VectorXcd &exp_ikR, 
        const int &lower_band_index, // counting from 1.
        const int &upper_band_index, // counting from 1, upper_band_index >= lower_band_index
        VectorXd &eigenvalues, 
        MatrixXcd &eigenvectors
    );

    static void get_eigenvalues(
        base_data &Base_Data, 
        const MatrixXd &k_direct_coor, 
        std::vector<VectorXd> &eigenvalues
    );

    static void get_eigenvalues_range(
        base_data &Base_Data, 
        const MatrixXd &k_direct_coor, 
        const int &lower_band_index, // counting from 1.
        const int &upper_band_index, // counting from 1, upper_band_index >= lower_band_index
        std::vector<VectorXd> &eigenvalues
    );
    
    static void get_eigenvalues_eigenvectors(
        base_data &Base_Data, 
        const MatrixXd &k_direct_coor, 
        std::vector<VectorXd> &eigenvalues, 
        std::vector<MatrixXcd> &eigenvectors
    );

    static void get_eigenvalues_eigenvectors_range(
        base_data &Base_Data, 
        const MatrixXd &k_direct_coor, 
        const int &lower_band_index, // counting from 1.
        const int &upper_band_index, // counting from 1, upper_band_index >= lower_band_index
        std::vector<VectorXd> &eigenvalues, 
        std::vector<MatrixXcd> &eigenvectors
    );

    // ARPACK shift-invert solver: find nev eigenvalues of H*v = lambda*S*v
    // closest to sigma (target energy, e.g. Fermi level).
    // Returns false and falls back gracefully if ARPACK fails to converge.
    static bool get_eigenvalues_arpack_1k(
        base_data &Base_Data,
        const VectorXcd &exp_ikR,
        const int &nev,
        const double &sigma,
        const int &ncv,
        const double &tol,
        const int &maxiter,
        VectorXd &eigenvalues
    );

    static bool get_eigenvalues_eigenvectors_arpack_1k(
        base_data &Base_Data,
        const VectorXcd &exp_ikR,
        const int &nev,
        const double &sigma,
        const int &ncv,
        const double &tol,
        const int &maxiter,
        VectorXd &eigenvalues,
        MatrixXcd &eigenvectors
    );
};

#endif