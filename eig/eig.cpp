#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <assert.h>
#include <time.h>
using namespace std;

const double pi = 3.14159265358979323846;
const double eps = 1e-8;

// random number generation by mersenne twister
random_device rd;
mt19937 generator(rd());
normal_distribution<double> distribution(0.0, 1.0);

struct Matrix {
    vector < vector <double> > val;
    Matrix(int n = 1, int m = 1) { val.clear(); val.resize(n, vector<double>(m)); }
    Matrix(int n, int m, double x) { val.clear(); val.resize(n, vector<double>(m, x)); }
    void init(int n, int m, double x = 0) { val.clear(); val.resize(n, vector<double>(m, x)); }
    void resize(int n, int m, double x = 0) { val.resize(n); for (int i = 0; i < n; i++) val[i].resize(m, x); }
    int size() { return val.size(); }
    inline vector <double> &operator[](int i) { return val[i]; }
    friend ostream &operator<<(ostream &s, Matrix &M) {
        for (int i = 0; i < M.size(); i++) {
            for (int j = 0; j < M[0].size(); j++) s << M[i][j] << " ";
            s << "\n";
        }
        return s;
    }
};

Matrix mul(Matrix &A, Matrix &B) {
    size_t sz = A[0].size();
    assert(sz == B.size());

    size_t row = A.size();
    size_t column = B[0].size();

    Matrix R(row, column, 0.0);
    for (size_t i = 0; i < row; i++)
        for (size_t k = 0; k < sz; k++)
            for (size_t j = 0; j < column; j++)
                R[i][j] += A[i][k] * B[k][j];

    return R;
}

Matrix add(Matrix &A, Matrix &B) {
    size_t row = A.size();
    size_t column = A[0].size();
    assert(row == B.size());
    assert(column == B[0].size());

    Matrix R(row, column, 0);
    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < column; j++)
            R[i][j] = A[i][j] + B[i][j];

    return R;
}

vector <double> mul(Matrix &A, vector <double> &v) {
    size_t sz = A[0].size();
    assert(sz == v.size());

    vector <double> r(A.size(), 0.0);
    for (size_t i = 0, row = A.size(); i < row; i++)
        for (size_t j = 0; j < sz; j++)
            r[i] += A[i][j] * v[j];

    return r;
}

vector <double> add(vector <double> &v1, vector <double> &v2) {
    size_t sz = v1.size();
    assert(sz == v2.size());

    vector <double> r(sz);
    for (size_t i = 0; i < sz; i++)
        r[i] = v1[i] + v2[i];

    return r;
}

vector <double> sub(vector <double> &v1, vector <double> &v2) {
    size_t sz = v1.size();
    assert(sz == v2.size());

    vector <double> r(sz);
    for (size_t i = 0; i < sz; i++)
        r[i] = v1[i] - v2[i];

    return r;
}

Matrix getCoFactor(Matrix &A, int p, int q) {
    // 正方行列のみ通す
    size_t sz = A.size();
    assert(sz == A[0].size());
    assert(sz > 0);

    Matrix co_factor(sz-1, sz-1);
    int i = 0, j = 0;
    for (size_t row = 0; row < sz; row++) {
        for (size_t col = 0; col < sz; col++) {
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
    // 正方行列のみ通す
    size_t sz = A.size();
    assert(sz == A[0].size());

    if (sz == 1) return A[0][0];

    double det = 0;
    for (size_t i = 0; i < sz; i++) {
        Matrix co_factor = getCoFactor(A, 0, i);
        int sign = i % 2 == 0 ? 1 : -1;
        det += sign * A[0][i] * determinant(co_factor);
    }

    return det;
}

Matrix adjoint(Matrix &A) {
    // 正方行列のみ通す
    size_t sz = A.size();
    assert(sz == A[0].size());

    Matrix adj(sz, sz);
    for (size_t j = 0; j < sz; j++) {
        for (size_t i = 0; i < sz; i++) {
            Matrix co_factor = getCoFactor(A, i, j);
            int sign = (i + j) % 2 == 0 ? 1 : -1;
            adj[j][i] = sign * determinant(co_factor);
        }
    }

    return adj;
}

Matrix Inverse(Matrix &A) {
    // 正方行列のみ通す
    size_t sz = A.size();
    assert(sz == A[0].size());

    Matrix inv(sz, sz);

    // 正則行列のみ通す
    double det = determinant(A);
    assert(abs(det - eps) > 0);

    Matrix adj = adjoint(A);

    for (size_t i = 0; i < sz; i++)
        for (size_t j = 0; j < sz; j++)
            inv[i][j] = adj[i][j] / det;

    return inv;
}

Matrix Transpose(Matrix &A) {
    size_t row = A.size();
    size_t column = A[0].size();
    Matrix R(column, row);
    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < column; j++)
            R[j][i] = A[i][j];

    return R;
}

Matrix Transpose(vector <double> &v) {
    size_t sz = v.size();
    Matrix R(1, sz);
    for (size_t i = 0; i < sz; i++)
        R[0][i] = v[i];

    return R;
}

void MatrixTest() {
    cout << "----------------------------\n";
    cout << "Test Matrix Starts!!\n";

    for (int t = 0; t < 10; t++) {
        int n = 5;
        Matrix m(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i][j] = distribution(generator);
            }
        }

        double d = determinant(m);
        if (d == 0) continue;

        Matrix inv = Inverse(m);
        Matrix I = mul(m, inv);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    assert(abs(I[i][j] - 1.0) < eps);
                } else {
                    assert(abs(I[i][j]) < eps);
                }
            }
        }
    }

    cout << "Test Matrix Inverse Passed!!\n";
    cout << "----------------------------\n";
}

struct EIG {
    Matrix A, TransA;

    vector <double> theta_u;
    Matrix theta_sigma;
    Matrix InvThetaSigma;
    double thetaPdfDenominator;

    vector <double> epsilon_u;
    Matrix epsilon_sigma;
    Matrix InvEpsilonSigma;
    double epsilonPdfDenominator;

    Matrix LaplaceUApproximation;
    Matrix LaplaceSigmaApproximation;

    int maxLevel;
    double init_M; // level 0 でのinner sample数

    int rand_index;
    vector <double> N, rand, P, P2, aveP, varP, Z, Z2, aveZ, varZ;

    void init() {
        A.resize(3, 2);
        A[0][0] = 1.0;
        A[0][1] = A[1][0] = 2.0;
        A[1][1] = A[2][0] = 3.0;
        A[2][1] = 4.0;

        TransA = Transpose(A);

        theta_u.resize(2, 0.0);
        theta_u[0] = 1.0;

        theta_sigma.resize(2, 2);
        theta_sigma[0][0] = theta_sigma[1][1] = 2.0;
        theta_sigma[0][1] = theta_sigma[1][0] = -1.0;

        InvThetaSigma = Inverse(theta_sigma);
        thetaPdfDenominator = 2.0 * pi * sqrt(determinant(theta_sigma));

        epsilon_u.resize(3, 0.0);

        epsilon_sigma.resize(3, 3, 0.0);
        epsilon_sigma[0][0] = epsilon_sigma[1][1] = epsilon_sigma[2][2] = 0.1;
        epsilon_sigma[0][1] = epsilon_sigma[1][0] = epsilon_sigma[2][1] = epsilon_sigma[1][2] = -0.05;

        InvEpsilonSigma = Inverse(epsilon_sigma);
        epsilonPdfDenominator = pow(2.0 * pi, 1.5) * sqrt(determinant(epsilon_sigma));

        Matrix tmp1 = mul(TransA, InvEpsilonSigma);
        Matrix tmp2 = mul(tmp1, A);
        Matrix tmp3 = add(tmp2, InvThetaSigma);
        LaplaceSigmaApproximation = Inverse(tmp3);
        LaplaceUApproximation = mul(LaplaceSigmaApproximation, tmp1);

        maxLevel = 20;
        init_M = 2.0;

        N.resize(maxLevel, 0.0);
        P.resize(maxLevel, 0.0);
        P2.resize(maxLevel, 0.0);
        aveP.resize(maxLevel, 0.0);
        varP.resize(maxLevel, 0.0);
        Z.resize(maxLevel, 0.0);
        Z2.resize(maxLevel, 0.0);
        aveZ.resize(maxLevel, 0.0);
        varZ.resize(maxLevel, 0.0);
    }

    void rand_init(int L, double outer) {
        rand_index = 0;

        int count = (4 + init_M * pow(2, L+2)) * outer;
        rand.resize(count);

        int c = 0;
        while (c < count) {
            rand[c] = distribution(generator);
            rand[c+1] = distribution(generator);
            rand[c+2] = distribution(generator);
            rand[c+3] = distribution(generator);
            c += 4;
        }
    }

    double thetaPdf(vector <double> &x) {
        size_t sz = theta_u.size();
        assert(x.size() == sz);

        vector <double> tmp1 = sub(x, theta_u);
        Matrix tmp2 = Transpose(tmp1);
        Matrix tmp3 = mul(tmp2, InvThetaSigma);
        vector <double> tmp4 = mul(tmp3, tmp1);

        double tmp5 = exp(- tmp4[0] / 2.0);

        return tmp5 / thetaPdfDenominator;
    }

    double epsilonPdf(vector <double> &x) {
        size_t sz = epsilon_u.size();
        assert(x.size() == sz);

        vector <double> tmp1 = sub(x, epsilon_u);
        Matrix tmp2 = Transpose(tmp1);
        Matrix tmp3 = mul(tmp2, InvEpsilonSigma);
        vector <double> tmp4 = mul(tmp3, tmp1);

        double tmp5 = exp(- tmp4[0] / 2.0);

        return tmp5 / epsilonPdfDenominator;
    }
};

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

vector <double> rand_multinormal(EIG &eig, vector <double> &u, Matrix &Sigma) {
    size_t sz = u.size();
    assert(Sigma.size() == sz);
    assert(Sigma[0].size() == sz);

    Matrix Q = Cholesky(Sigma);
    vector <double> rand(sz);
    vector <double> ret(sz);
    for (size_t i = 0; i < sz; i++) {
        // rand[i] = eig.rand[eig.rand_index++];
        rand[i] = distribution(generator);
        for (size_t j = 0; j <= i; j++)
            ret[i] += Q[i][j] * rand[j];

        ret[i] += u[i];
    }

    return ret;
}

double pdf(vector <double> &x, vector <double> &u, Matrix &sigma) {
    size_t sz = x.size();
    assert(u.size() == sz);
    assert(sigma.size() == sz);
    assert(sigma[0].size() == sz);

    // f(x) = exp(-1/2 * transpose(x - u) * Σ^-1 * (x - u)) / ((2π)^(n/2) * sqrt(|Σ|))
    vector <double> tmp1 = sub(x, u);
    Matrix tmp2 = Transpose(tmp1);
    Matrix tmp3 = Inverse(sigma);
    Matrix tmp4 = mul(tmp2, tmp3);
    vector <double> tmp5 = mul(tmp4, tmp1);

    double tmp6 = exp(- tmp5[0] / 2.0);

    double tmp7 = pow(2.0 * pi, (double)sz / 2.0) * sqrt(determinant(sigma));

    return tmp6 / tmp7;
}

void pdfTest() {
    cout << "----------------------------\n";
    cout << "Test PDF Starts!!\n";

    // εの平均
    vector <double> epsilon_u(3, 0.0);

    // εの分散共分散行列
    Matrix epsilon_sigma(3, 3, 0.0);
    epsilon_sigma[0][0] = epsilon_sigma[1][1] = epsilon_sigma[2][2] = 0.1;
    epsilon_sigma[0][1] = epsilon_sigma[1][0] = epsilon_sigma[2][1] = epsilon_sigma[1][2] = -0.05;

    vector <double> v(3, 0);
    double p = pdf(v, epsilon_u, epsilon_sigma);

    assert(abs(p - 2.839521721752) < eps);

    cout << "Test PDF Passed!!\n";
    cout << "----------------------------\n";
}

double ImportanceSampling(EIG &eig, vector <double> &Y, vector <double> &theta) {
    vector <double> tmp1 = mul(eig.A, theta);
    vector <double> tmp2 = sub(Y, tmp1);
    vector <double> tmp3 = mul(eig.LaplaceUApproximation, tmp2);
    vector <double> u_approximation = sub(theta, tmp3);
    Matrix sigma_approximation = eig.LaplaceSigmaApproximation;

    vector <double> new_theta = rand_multinormal(eig, u_approximation, sigma_approximation);
    tmp1 = mul(eig.A, new_theta);
    tmp2 = sub(Y, tmp1);
    double p_y_given_theta = eig.epsilonPdf(tmp2);
    double p_theta = eig.thetaPdf(theta);
    double q = pdf(new_theta, u_approximation, sigma_approximation);

    // cout << p_y_given_theta << " " << p_theta << " " << q << '\n';
    return p_y_given_theta * p_theta / q;
}

void eig_calc(EIG &eig, int l, double M) {
    vector <double> theta = rand_multinormal(eig, eig.theta_u, eig.theta_sigma);
    vector <double> Atheta = mul(eig.A, theta);
    vector <double> epsilon = rand_multinormal(eig, eig.epsilon_u, eig.epsilon_sigma);
    vector <double> Y = add(Atheta, epsilon);

    // log(p(Y|θ))は、εを生成し、それがεの分布からどのくらいの確率で生成されるかを求めれば良い
    double log_p_y_theta = log(pdf(epsilon, eig.epsilon_u, eig.epsilon_sigma));

    if (l == 0) {
        double sum = 0;
        for (int m = 0; m < M; m++) {
            sum += ImportanceSampling(eig, Y, theta);
        }

        double p = log_p_y_theta - log(sum / M);
        eig.P[l] += p;
        eig.P2[l] += p * p;
        eig.Z[l] += p;
        eig.Z2[l] += p * p;
    } else {
        double sum_a = 0, sum_b = 0;
        for (int m = 0; m < M / 2; m++) {
            sum_a += ImportanceSampling(eig, Y, theta);
            sum_b += ImportanceSampling(eig, Y, theta);
        }

        double sum = sum_a + sum_b;

        double p = log_p_y_theta - log(sum / M);
        eig.P[l] += p;
        eig.P2[l] += p * p;

        double z = (log(sum_a * 2.0 / M) + log(sum_b * 2.0 / M)) / 2.0 - log(sum / M);
        eig.Z[l] += z;
        eig.Z2[l] += z * z;
    }
}

void calc(EIG &eig, int l, double M) {
    for (int n = 0; n < eig.N[l]; n++) {
        eig_calc(eig, l, M);
    }

    eig.aveP[l] = abs(eig.P[l] / eig.N[l]);
    eig.varP[l] = eig.P2[l] / eig.N[l] - eig.aveP[l] * eig.aveP[l];

    eig.aveZ[l] = abs(eig.Z[l] / eig.N[l]);
    eig.varZ[l] = eig.Z2[l] / eig.N[l] - eig.aveZ[l] * eig.aveZ[l];

    cout << log2(eig.aveP[l]) << " " << log2(eig.aveZ[l]) << " " << log2(eig.varP[l]) << " " << log2(eig.varZ[l]) << '\n';
}

void estimate_alpha_beta(EIG &eig, int L, double n) {
    fill(eig.N.begin(), eig.N.end(), n);

    cout << "aveP    aveZ    varP    varZ\n";

    double M = eig.init_M;
    for (int l = 0; l <= L; l++) {
        calc(eig, l, M);
        M *= 2;
    }
}

int main() {
    MatrixTest();
    pdfTest();

    clock_t start = clock();

    EIG eig;
    eig.init();

    int L = 10;
    double outer = 2000.0;

    // eig.rand_init(L, outer);
    estimate_alpha_beta(eig, L, outer);

    clock_t end = clock();
    cout << "time = " << (double)(end - start) / CLOCKS_PER_SEC << " (sec)\n";
}