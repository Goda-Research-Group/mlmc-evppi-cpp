
#include "matrix.hpp"

random_device _rd;
mt19937 _generator(_rd());
normal_distribution<double> dist(0.0, 1.0);

const double INF = 1e30;

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

vector<double> rand_multinormal(vector<double> &u, Matrix &sigma_cholesky, vector<double> &pre) {
    size_t sz = u.size();
    vector<double> rand(sz);
    vector<double> ret(sz);
    for (size_t i = 0; i < sz; i++) {
        if (pre[i] < INF) {
            rand[i] = pre[i];
        } else {
            rand[i] = dist(_generator);
        }

        for (size_t j = 0; j <= i; j++)
            ret[i] += sigma_cholesky[i][j] * rand[j];
        ret[i] += u[i];
    }
    return ret;
}