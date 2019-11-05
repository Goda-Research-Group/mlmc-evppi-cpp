
#include "matrix.hpp"

Matrix Cholesky(Matrix &A) {
    int sz = A.size();
    Matrix Q(sz, sz);
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < i; j++) {
            Q[i][j] = A[i][j];
            for (int k = 0; k < j; k++)
                Q[i][j] -= Q[i][k] * Q[j][k];
            Q[i][j] /= Q[j][j];
        }
        Q[i][i] = A[i][i];
        for (int k = 0; k < i; k++)
            Q[i][i] -= Q[i][k] * Q[i][k];
        Q[i][i] = sqrt(Q[i][i]);
    }
    return Q;
}

Matrix operator - (Matrix &A, Matrix &B) {
    Matrix R(A.size(), A[0].size());
    for (int i = 0; i < A.size(); ++i)
        for (int j = 0; j < (int)A[0].size(); ++j)
            R[i][j] = A[i][j] - B[i][j];
    return R;
}

Matrix operator * (Matrix &A, Matrix &B) {
    Matrix R(A.size(), B[0].size());
    for (int i = 0; i < A.size(); ++i)
        for (int j = 0; j < (int)B[0].size(); ++j)
            for (int k = 0; k < B.size(); ++k)
                R[i][j] += A[i][k] * B[k][j];
    return R;
}

Matrix getCoFactor(Matrix &A, const int p, const int q) {
    int sz = A.size();
    Matrix co_factor(sz - 1, sz - 1);
    int i = 0, j = 0;
    for (int row = 0; row < sz; row++) {
        for (int col = 0; col < sz; col++) {
            if (row != p && col != q) {
                co_factor[i][j] = A[row][col];
                j++;
                if (j == sz - 1) {
                    j = 0;
                    i++;
                }
            }
        }
    }
    return co_factor;
}

double determinant(Matrix &A) {
    int sz = A.size();
    if (sz == 1) return A[0][0];
    double det = 0;
    for (int i = 0; i < sz; i++) {
        Matrix co_factor = getCoFactor(A, 0, i);
        int sign = i % 2 == 0 ? 1 : -1;
        det += sign * A[0][i] * determinant(co_factor);
    }
    return det;
}

Matrix adjoint(Matrix &A) {
    int sz = A.size();
    Matrix adj(sz, sz);
    for (int j = 0; j < sz; j++) {
        for (int i = 0; i < sz; i++) {
            Matrix co_factor = getCoFactor(A, i, j);
            int sign = (i + j) % 2 == 0 ? 1 : -1;
            adj[j][i] = sign * determinant(co_factor);
        }
    }
    return adj;
}

Matrix Inverse(Matrix &A) {
    int sz = A.size();
    Matrix inv(sz, sz);
    double det = determinant(A);
    Matrix adj = adjoint(A);
    for (int i = 0; i < sz; i++)
        for (int j = 0; j < sz; j++)
            inv[i][j] = adj[i][j] / det;
    return inv;
}

vector<double> rand_multinormal(vector<double> &u, Matrix &sigma_cholesky, vector<double> &rand) {
    int sz = u.size();
    vector<double> ret(sz);
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j <= i; j++)
            ret[i] += sigma_cholesky[i][j] * rand[j];
        ret[i] += u[i];
    }
    return ret;
}