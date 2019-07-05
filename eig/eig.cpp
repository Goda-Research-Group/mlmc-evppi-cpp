#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <assert.h>
#include <chrono>
#include <thread>
#include <mutex>
using namespace std;
typedef vector <double> vec;

const double pi = 3.14159265358979323846;
const double eps = 1e-8;

// random number generation by mersenne twister
random_device rd;
mt19937 generator(rd());
normal_distribution<double> distribution(0.0, 1.0);

struct Matrix {
    vector < vec > val;
    Matrix(int n = 1, int m = 1) { val.clear(); val.resize(n, vec(m)); }
    Matrix(int n, int m, double x) { val.clear(); val.resize(n, vec(m, x)); }
    void init(int n, int m, double x = 0) { val.clear(); val.resize(n, vec(m, x)); }
    void resize(int n, int m, double x = 0) { val.resize(n); for (int i = 0; i < n; i++) val[i].resize(m, x); }
    int size() { return val.size(); }
    inline vec &operator[](int i) { return val[i]; }
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

vec mul(Matrix &A, const vec &v) {
    size_t sz = A[0].size();
    assert(sz == v.size());

    vec r(A.size(), 0.0);
    for (size_t i = 0, row = A.size(); i < row; i++)
        for (size_t j = 0; j < sz; j++)
            r[i] += A[i][j] * v[j];

    return r;
}

vec add(const vec &v1, const vec &v2) {
    size_t sz = v1.size();
    assert(sz == v2.size());

    vec r(sz);
    for (size_t i = 0; i < sz; i++)
        r[i] = v1[i] + v2[i];

    return r;
}

vec sub(const vec &v1, const vec &v2) {
    size_t sz = v1.size();
    assert(sz == v2.size());

    vec r(sz);
    for (size_t i = 0; i < sz; i++)
        r[i] = v1[i] - v2[i];

    return r;
}

Matrix getCoFactor(Matrix &A, const int p, const int q) {
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

Matrix Transpose(const vec &v) {
    size_t sz = v.size();
    Matrix R(1, sz);
    for (size_t i = 0; i < sz; i++)
        R[0][i] = v[i];

    return R;
}

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

struct Eig {
    Matrix A, trans_A;

    vec theta_u;
    Matrix theta_sigma;
    Matrix inv_theta_sigma;
    Matrix theta_cholesky;
    double theta_pdf_denominator;

    vec epsilon_u;
    Matrix epsilon_sigma;
    Matrix inv_epsilon_sigma;
    Matrix epsilon_cholesky;
    double epsilon_pdf_denominator;

    Matrix laplace_u_approximation;
    Matrix laplace_sigma_approximation;
    Matrix laplace_cholesky;

    int max_level;
    int num_thread;
    mutex mtx;

    // level 0 でのinner sample数
    double init_M;
    double alpha, beta, gamma;

    vec N, P, P2, aveP, varP, Z, Z2, aveZ, varZ;

    void init() {
        A.resize(3, 2);
        A[0][0] = 1.0;
        A[0][1] = A[1][0] = 2.0;
        A[1][1] = A[2][0] = 3.0;
        A[2][1] = 4.0;

        trans_A = Transpose(A);

        theta_u.resize(2, 0.0);
        theta_u[0] = 1.0;

        theta_sigma.resize(2, 2);
        theta_sigma[0][0] = theta_sigma[1][1] = 2.0;
        theta_sigma[0][1] = theta_sigma[1][0] = -1.0;
        theta_cholesky = Cholesky(theta_sigma);

        inv_theta_sigma = Inverse(theta_sigma);
        theta_pdf_denominator = 2.0 * pi * sqrt(determinant(theta_sigma));

        epsilon_u.resize(3, 0.0);

        epsilon_sigma.resize(3, 3, 0.0);
        epsilon_sigma[0][0] = epsilon_sigma[1][1] = epsilon_sigma[2][2] = 0.1;
        epsilon_sigma[0][1] = epsilon_sigma[1][0] = epsilon_sigma[2][1] = epsilon_sigma[1][2] = -0.05;
        epsilon_cholesky = Cholesky(epsilon_sigma);

        inv_epsilon_sigma = Inverse(epsilon_sigma);
        epsilon_pdf_denominator = pow(2.0 * pi, 1.5) * sqrt(determinant(epsilon_sigma));

        Matrix tmp1 = mul(trans_A, inv_epsilon_sigma);
        Matrix tmp2 = mul(tmp1, A);
        Matrix tmp3 = add(tmp2, inv_theta_sigma);
        laplace_sigma_approximation = Inverse(tmp3);
        laplace_u_approximation = mul(laplace_sigma_approximation, tmp1);
        laplace_cholesky = Cholesky(laplace_sigma_approximation);

        max_level = 20;
        num_thread = thread::hardware_concurrency() / 2;
        init_M = 2.0;

        N.resize(max_level, 0.0);
        P.resize(max_level, 0.0);
        P2.resize(max_level, 0.0);
        aveP.resize(max_level, 0.0);
        varP.resize(max_level, 0.0);
        Z.resize(max_level, 0.0);
        Z2.resize(max_level, 0.0);
        aveZ.resize(max_level, 0.0);
        varZ.resize(max_level, 0.0);
    }

    double theta_pdf(const vec &x) {
        size_t sz = theta_u.size();
        assert(x.size() == sz);

        vec tmp1 = sub(x, theta_u);
        Matrix tmp2 = Transpose(tmp1);
        Matrix tmp3 = mul(tmp2, inv_theta_sigma);
        vec tmp4 = mul(tmp3, tmp1);

        double tmp5 = exp(- tmp4[0] / 2.0);

        return tmp5 / theta_pdf_denominator;
    }

    double epsilon_pdf(const vec &x) {
        size_t sz = epsilon_u.size();
        assert(x.size() == sz);

        vec tmp1 = sub(x, epsilon_u);
        Matrix tmp2 = Transpose(tmp1);
        Matrix tmp3 = mul(tmp2, inv_epsilon_sigma);
        vec tmp4 = mul(tmp3, tmp1);

        double tmp5 = exp(- tmp4[0] / 2.0);

        return tmp5 / epsilon_pdf_denominator;
    }

    vec rand_theta() {
        vec rand(2);
        vec ret(2);
        for (int i = 0; i < 2; i++) {
            rand[i] = distribution(generator);
            for (int j = 0; j <= i; j++)
                ret[i] += theta_cholesky[i][j] * rand[j];
            ret[i] += theta_u[i];
        }
        return ret;
    }

    vec rand_epsilon() {
        vec rand(3);
        vec ret(3);
        for (int i = 0; i < 3; i++) {
            rand[i] = distribution(generator);
            for (int j = 0; j <= i; j++)
                ret[i] += epsilon_cholesky[i][j] * rand[j];
            ret[i] += epsilon_u[i];
        }
        return ret;
    }

    vec rand_laplace(const vec &u) {
        vec rand(2);
        vec ret(2);
        for (int i = 0; i < 2; i++) {
            rand[i] = distribution(generator);
            for (int j = 0; j <= i; j++)
                ret[i] += laplace_cholesky[i][j] * rand[j];
            ret[i] += u[i];
        }
        return ret;
    }
};

vec rand_multinormal(Eig &eig, const vec &u, Matrix &sigma) {
    size_t sz = u.size();
    assert(sigma.size() == sz);
    assert(sigma[0].size() == sz);

    Matrix Q = Cholesky(sigma);
    vec rand(sz);
    vec ret(sz);
    for (size_t i = 0; i < sz; i++) {
        rand[i] = distribution(generator);
        for (size_t j = 0; j <= i; j++)
            ret[i] += Q[i][j] * rand[j];

        ret[i] += u[i];
    }

    return ret;
}

double pdf(const vec &x, const vec &u, Matrix &sigma) {
    size_t sz = x.size();
    assert(u.size() == sz);
    assert(sigma.size() == sz);
    assert(sigma[0].size() == sz);

    // f(x) = exp(-1/2 * transpose(x - u) * Σ^-1 * (x - u)) / ((2π)^(n/2) * sqrt(|Σ|))
    vec tmp1 = sub(x, u);
    Matrix tmp2 = Transpose(tmp1);
    Matrix tmp3 = Inverse(sigma);
    Matrix tmp4 = mul(tmp2, tmp3);
    vec tmp5 = mul(tmp4, tmp1);

    double tmp6 = exp(- tmp5[0] / 2.0);

    double tmp7 = pow(2.0 * pi, (double)sz / 2.0) * sqrt(determinant(sigma));

    return tmp6 / tmp7;
}

void pdfTest() {
    cout << "----------------------------\n";
    cout << "Test PDF Starts!!\n";

    // εの平均
    vec epsilon_u(3, 0.0);

    // εの分散共分散行列
    Matrix epsilon_sigma(3, 3, 0.0);
    epsilon_sigma[0][0] = epsilon_sigma[1][1] = epsilon_sigma[2][2] = 0.1;
    epsilon_sigma[0][1] = epsilon_sigma[1][0] = epsilon_sigma[2][1] = epsilon_sigma[1][2] = -0.05;

    vec v(3, 0);
    double p = pdf(v, epsilon_u, epsilon_sigma);

    assert(abs(p - 2.839521721752) < eps);

    cout << "Test PDF Passed!!\n";
    cout << "----------------------------\n";
}

double importance_sampling(Eig &eig, const vec &Y, const vec &u_approximation) {
    vec new_theta = eig.rand_laplace(u_approximation);
    vec tmp = mul(eig.A, new_theta);
    tmp = sub(Y, tmp);
    double p_y_given_theta = eig.epsilon_pdf(tmp);
    double q = pdf(new_theta, u_approximation, eig.laplace_sigma_approximation);

    return p_y_given_theta / q;
}

void eig_calc(Eig &eig, const int l, const double M) {
    vec theta = eig.rand_theta();
    double p_theta = eig.theta_pdf(theta);

    vec A_theta = mul(eig.A, theta);
    vec epsilon = eig.rand_epsilon();
    vec Y = add(A_theta, epsilon);

    vec tmp = mul(eig.laplace_u_approximation, epsilon);
    vec u_approximation = add(theta, tmp);

    // log(p(Y|θ))は、εを生成し、それがεの分布からどのくらいの確率で生成されるかを求めれば良い
    double log_p_y_theta = log(pdf(epsilon, eig.epsilon_u, eig.epsilon_sigma));

    if (l == 0) {
        double sum = 0;
        for (int m = 0; m < M; m++) {
            sum += importance_sampling(eig, Y, u_approximation) * p_theta;
        }

        double p = log_p_y_theta - log(sum / M);

        eig.mtx.lock();
        eig.P[l] += p;
        eig.P2[l] += p * p;
        eig.Z[l] += p;
        eig.Z2[l] += p * p;
        eig.mtx.unlock();
    } else {
        double sum_a = 0, sum_b = 0;
        for (int m = 0; m < M / 2; m++) {
            sum_a += importance_sampling(eig, Y, u_approximation) * p_theta;
            sum_b += importance_sampling(eig, Y, u_approximation) * p_theta;
        }

        double sum = sum_a + sum_b;

        double p = log_p_y_theta - log(sum / M);
        double z = (log(sum_a * 2.0 / M) + log(sum_b * 2.0 / M)) / 2.0 - log(sum / M);

        eig.mtx.lock();
        eig.P[l] += p;
        eig.P2[l] += p * p;
        eig.Z[l] += z;
        eig.Z2[l] += z * z;
        eig.mtx.unlock();
    }
}

void calc(Eig &eig, const int l, const double M) {
    for (int n = 0; n < eig.N[l] / eig.num_thread; n++) {
        vector <thread> th(eig.num_thread);
        for (int j = 0; j < eig.num_thread; j++) th[j] = thread(eig_calc, ref(eig), l, M);
        for (int j = 0; j < eig.num_thread; j++) th[j].join();
    }
}

double regression(const vector <double> &x, const vector <double> &y) {
    double sz = x.size();
    double sum_x = 0.0, sum_y = 0.0, sum_x_x = 0.0, sum_x_y = 0.0;
    for (int l = 0; l < sz; l++) {
        sum_x += x[l];
        sum_y += y[l];
        sum_x_x += x[l] * x[l];
        sum_x_y += x[l] * y[l];
    }

    return (sz * sum_x_y - sum_x * sum_y) / (sz * sum_x_x - sum_x * sum_x);
}

void estimate_alpha_beta(Eig &eig, const int L, const double n) {
    fill(eig.N.begin(), eig.N.end(), n);

    cout << "----------------------------\n";
    cout << "aveP    aveZ    varP    varZ\n";

    double M = eig.init_M;
    for (int l = 0; l <= L; l++, M *= 2) {
        calc(eig, l, M);

        eig.aveP[l] = abs(eig.P[l] / eig.N[l]);
        eig.varP[l] = eig.P2[l] / eig.N[l] - eig.aveP[l] * eig.aveP[l];

        eig.aveZ[l] = abs(eig.Z[l] / eig.N[l]);
        eig.varZ[l] = eig.Z2[l] / eig.N[l] - eig.aveZ[l] * eig.aveZ[l];

        cout << log2(eig.aveP[l]) << " " << log2(eig.aveZ[l]) << " " << log2(eig.varP[l]) << " " << log2(eig.varZ[l]) << '\n';
    }

    cout << "----------------------------\n";
    cout << "linear regression\n";

    vector<double> x(L);
    vector<double> y(L);

    // liner regression for alpha
    for (int l = 1; l <= L; l++) {
        x[l - 1] = l;
        y[l - 1] = -log2(eig.aveZ[l]);
    }
    eig.alpha = regression(x, y);
    cout << "alpha = " << eig.alpha << "\n";

    // liner regression for beta
    for (int l = 1; l <= L; l++) {
        x[l - 1] = l;
        y[l - 1] = -log2(eig.varZ[l]);
    }
    eig.beta = regression(x, y);
    cout << "beta =  " << eig.beta << "\n";

    cout << "----------------------------\n";
}

int main() {
    MatrixTest();
    pdfTest();

    auto start = chrono::high_resolution_clock::now();

    Eig eig;
    eig.init();

    int L = 10;
    double outer = 2000.0;

    estimate_alpha_beta(eig, L, outer);

    auto end = chrono::high_resolution_clock::now();
    auto msec = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "total time = " << msec / 1000.0 << " (sec)\n";
}