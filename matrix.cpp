
#include "matrix.hpp"

Matrix Cholesky(Matrix &A) {
    size_t sz = A.size();
    Matrix Q(sz, sz);
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = 0; j < i; j++) {
            Q[i][j] = A[i][j];
            for (size_t k = 0; k < j; k++)
                Q[i][j] -= Q[i][k] * Q[j][k];
            Q[i][j] /= Q[j][j];
        }
        Q[i][i] = A[i][i];
        for (size_t k = 0; k < i; k++)
            Q[i][i] -= Q[i][k] * Q[i][k];
        Q[i][i] = sqrt(Q[i][i]);
    }
    return Q;
}

vector<double> rand_multinormal(vector<double> &u, Matrix &sigma_cholesky, vector<double> &rand) {
    size_t sz = u.size();
    vector<double> ret(sz);
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = 0; j <= i; j++)
            ret[i] += sigma_cholesky[i][j] * rand[j];
        ret[i] += u[i];
    }
    return ret;
}