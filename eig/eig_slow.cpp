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

const int Lmax = 20;
double alpha, beta;

vector <double> N(Lmax);
vector <double> P(Lmax);
vector <double> P2(Lmax);
vector <double> aveP(Lmax);
vector <double> varP(Lmax);
vector <double> Z(Lmax);
vector <double> Z2(Lmax);
vector <double> aveZ(Lmax);
vector <double> varZ(Lmax);

template<class T> struct Matrix {
    vector < vector <double> > val;
    Matrix(int n = 1, int m = 1) { val.clear(); val.resize(n, vector<T>(m)); }
    Matrix(int n, int m, T x) { val.clear(); val.resize(n, vector<T>(m, x)); }
    void init(int n, int m, T x = 0) { val.clear(); val.resize(n, vector<T>(m, x)); }
    void resize(int n, int m, T x = 0) { val.resize(n); for (int i = 0; i < n; i++) val[i].resize(m, x); }
    int size() { return val.size(); }
    inline vector <T> &operator[](int i) { return val[i]; }
    friend ostream &operator<<(ostream &s, Matrix<T> M) {
        for (int i = 0; i < M.size(); i++) {
            for (int j = 0; j < M[0].size(); j++) s << M[i][j] << " ";
            s << "\n";
        }
        return s;
    }
};

template<class T> Matrix<T> operator * (Matrix<T> A, Matrix<T> B) {
    assert(A[0].size() == B.size());

    Matrix<T> R(A.size(), B[0].size(), 0);
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < B[0].size(); j++)
            for (int k = 0; k < B.size(); k++)
                R[i][j] += A[i][k] * B[k][j];

    return R;
}

template<class T> Matrix<T> operator + (Matrix<T> A, Matrix<T> B) {
    assert(A.size() == B.size());
    assert(A[0].size() == B[0].size());

    Matrix<T> R(A.size(), A[0].size());
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < A[0].size(); j++)
            R[i][j] = A[i][j] + B[i][j];

    return R;
}

template<class T> vector <T> operator * (Matrix<T> A, vector<T> B) {
    assert(A[0].size() == B.size());

    vector <T> v(A.size(), 0);
    for (int i = 0; i < A.size(); i++)
        for (int k = 0; k < B.size(); k++)
            v[i] += A[i][k] * B[k];

    return v;
}

template<class T> vector <T> operator + (vector<T> A, vector<T> B) {
    assert(A.size() == B.size());

    vector<T> v(A.size());
    for (int i = 0; i < A.size(); i++)
        v[i] = A[i] + B[i];

    return v;
}

template<class T> vector<T> operator - (vector<T> A, vector<T> B) {
    assert(A.size() == B.size());

    vector <T> v(A.size());
    for (int i = 0; i < A.size(); i++)
        v[i] = A[i] - B[i];

    return v;
}

template<class T> T operator * (vector<T> A, vector<T> B) {
    assert(A.size() == B.size());

    T ret = 0;
    for (int i = 0; i < A.size(); i++)
        ret += A[i] * B[i];

    return ret;
}

template<class T> Matrix<T> Cholesky(Matrix<T> A) {
    Matrix<T> Q(A.size(), A.size());
    for (int i = 0; i < A.size(); i++) {
        for (int j = 0; j < i; j++) {
            Q[i][j] = A[i][j];
            for (int k = 0; k < j; k++) Q[i][j] -= Q[i][k] * Q[j][k];
            Q[i][j] /= Q[j][j];
        }
        Q[i][i] = A[i][i];
        for (int k = 0; k < i; k++) Q[i][i] -= Q[i][k] * Q[i][k];
        Q[i][i] = sqrt(Q[i][i]);
    }
    return Q;
}

template<class T> Matrix<T> Transpose(Matrix<T> A) {
    Matrix<T> B(A[0].size(), A.size());
    for (int i = 0; i < A.size(); i++)
        for (int j = 0; j < A[0].size(); j++)
            B[j][i] = A[i][j];
    return B;
}

template<class T> Matrix<T> Transpose(vector<T> A) {
    Matrix<T> B(1, A.size());
    for (int i = 0; i < A.size(); i++) B[0][i] = A[i];
    return B;
}

template<class T> vector<T> rand_multinormal(vector <T> u, Matrix<T> Sigma) {
    int sz = u.size();
    assert(Sigma.size() == sz);
    assert(Sigma[0].size() == sz);

    Matrix<T> Q = Cholesky(Sigma);
    vector <double> rand(sz);
    vector <double> ret(sz);
    for (int i = 0; i < sz; i++) {
        rand[i] = distribution(generator);
        for (int j = 0; j <= i; j++) ret[i] += Q[i][j] * rand[j];
        ret[i] += u[i];
    }

    return ret;
}

template<class T> Matrix<T> getCoFactor(Matrix<T> A, int p, int q) {
    // 正方行列のみ通す
    int sz = A.size();
    assert(sz == A[0].size());

    Matrix<T> co_factor(sz-1, sz-1);
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

double determinant(Matrix<double> A) {
    // 正方行列のみ通す
    int sz = A.size();
    assert(sz == A[0].size());

    if (sz == 1) return A[0][0];

    double det = 0;
    for (int i = 0; i < sz; i++) {
        Matrix<double> co_factor = getCoFactor(A, 0, i);
        int sign = i % 2 == 0 ? 1 : -1;
        det += sign * A[0][i] * determinant(co_factor);
    }

    return det;
}

template<class T> Matrix<T> adjoint(Matrix<T> A) {
    // 正方行列のみ通す
    int sz = A.size();
    assert(sz == A[0].size());

    Matrix<T> adj(sz, sz);
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            Matrix<T> co_factor = getCoFactor(A, i, j);
            int sign = (i + j) % 2 == 0 ? 1 : -1;
            adj[j][i] = sign * determinant(co_factor);
        }
    }

    return adj;
}

template<class T> Matrix<T> Inverse(Matrix<T> A) {
    // 正方行列のみ通す
    int sz = A.size();
    assert(sz == A[0].size());

    Matrix<T> inv(sz, sz);

    // 正則行列のみ通す
    double det = determinant(A);
    assert(abs(det - eps) > 0);

    Matrix<T> adj = adjoint(A);

    for (int i = 0; i < sz; i++)
        for (int j = 0; j < sz; j++)
            inv[i][j] = adj[i][j] / det;

    return inv;
}

double pdf(vector <double> x, vector <double> u, Matrix <double> sigma) {
    int sz = x.size();
    assert(u.size() == sz);
    assert(sigma.size() == sz);
    assert(sigma[0].size() == sz);

    // f(x) = exp(-1/2 * transpose(x - u) * Σ^-1 * (x - u)) / ((2π)^(n/2) * sqrt(|Σ|))
    vector <double> tmp1 = Inverse(sigma) * (x - u);
    double tmp2 = (x - u) * tmp1;
    tmp2 = exp(- tmp2 / 2.0);
    double tmp3 = pow(2.0 * pi, (double)sz / 2.0) * sqrt(determinant(sigma));
    return tmp2 / tmp3;
}

Matrix <double> A(3, 2);
Matrix <double> InvA(2, 3);
vector <double> theta_u(2);
Matrix <double> theta_sigma(2, 2);
vector <double> epsilon_u(3, 0.0);
Matrix <double> epsilon_sigma(3, 3, 0.0);

double ImportanceSampling(vector <double> Y, vector <double> theta) {
    vector <double> u_approximation = theta - Inverse(Transpose(A) * Inverse(epsilon_sigma) * A + Inverse(theta_sigma)) * Transpose(A) * Inverse(epsilon_sigma) * (Y - A * theta);
    Matrix <double> sigma_approximation = Inverse(Transpose(A) * Inverse(epsilon_sigma) * A + Inverse(theta_sigma));

    vector <double> new_theta = rand_multinormal(u_approximation, sigma_approximation);
    double p_y_given_theta = pdf(Y - A * new_theta, epsilon_u, epsilon_sigma);
    double p_theta = pdf(theta, theta_u, theta_sigma);
    double q = pdf(new_theta, u_approximation, sigma_approximation);

    return p_y_given_theta * p_theta / q;
}

void eig(int l, double M) {
    // θを分布に従って生成
    vector <double> theta = rand_multinormal(theta_u, theta_sigma);

    // A * θ
    vector <double> Atheta = A * theta;

    // 分布に従ってεを生成
    vector <double> epsilon = rand_multinormal(epsilon_u, epsilon_sigma);

    // A, θ, εからYを導出
    vector <double> Y = Atheta + epsilon;

    // log(p(Y|θ))は、εを生成し、それがεの分布からどのくらいの確率で生成されるかを求めれば良い
    double log_p_y_theta = log(pdf(epsilon, epsilon_u, epsilon_sigma));

    if (l == 0) {
        double sum = 0;
        for (int m = 0; m < M; m++) {
            sum += ImportanceSampling(Y, theta);
        }

        double p = log_p_y_theta - log(sum / M);
        P[l] += p;
        P2[l] += p * p;
        Z[l] += p;
        Z2[l] += p * p;
    } else {
        double sum_a = 0, sum_b = 0;
        for (int m = 0; m < M / 2; m++) {
            sum_a += ImportanceSampling(Y, theta);
        }
        for (int m = M / 2; m < M; m++) {
            sum_b += ImportanceSampling(Y, theta);
        }

        double sum = sum_a + sum_b;

        double p = log_p_y_theta - log(sum / M);
        P[l] += p;
        P2[l] += p * p;

        double z = (log(sum_a * 2 / M) + log(sum_b * 2 / M)) / 2 - log(sum / M);
        Z[l] += z;
        Z2[l] += z * z;
    }
}

void calc(int l, double M) {
    for (int n = 0; n < N[l]; n++) {
        eig(l, M);
    }

    aveP[l] = abs(P[l] / N[l]);
    varP[l] = P2[l] / N[l] - aveP[l] * aveP[l];

    aveZ[l] = abs(Z[l] / N[l]);
    varZ[l] = Z2[l] / N[l] - aveZ[l] * aveZ[l];

    cout << log2(aveP[l]) << " " << log2(aveZ[l]) << " " << log2(varP[l]) << " " << log2(varZ[l]) << '\n';
}

void estimate_alpha_beta(int L, double outer_sample) {
    fill(N.begin(), N.end(), outer_sample);
    fill(P.begin(), P.end(), 0.0);
    fill(P2.begin(), P2.end(), 0.0);
    fill(Z.begin(), Z.end(), 0.0);
    fill(Z2.begin(), Z2.end(), 0.0);

    cout << "aveP    aveZ    varP    varZ\n";

    double M = 2;
    for (int l = 0; l <= L; l++) {
        calc(l, M);
        M *= 2;
    }

    // regressionする
}

template<class T> void printMatrix(Matrix<T> A) {
    for (int i = 0; i < A.size(); i++) {
        for (int j = 0; j < A[0].size(); j++) {
            cout << A[i][j] << " ";
        }
        cout << "\n";
    }
}

void MatrixTest() {
    cout << "----------------------------\n";
    cout << "Test Matrix Starts!!\n";

    for (int t = 0; t < 10; t++) {
        int n = 5;
        Matrix<double> m(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i][j] = distribution(generator);
            }
        }

        double d = determinant(m);
        if (d == 0) continue;

        Matrix<double> inv = Inverse(m);
        Matrix<double> I = m * inv;

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

void pdfTest() {
    cout << "----------------------------\n";
    cout << "Test PDF Starts!!\n";

    // εの平均
    vector <double> epsilon_u(3, 0.0);

    // εの分散共分散行列
    Matrix <double> epsilon_sigma(3, 3, 0.0);
    epsilon_sigma[0][0] = epsilon_sigma[1][1] = epsilon_sigma[2][2] = 0.1;
    epsilon_sigma[0][1] = epsilon_sigma[1][0] = epsilon_sigma[2][1] = epsilon_sigma[1][2] = -0.05;

    vector <double> v(3, 0);
    double p = pdf(v, epsilon_u, epsilon_sigma);

    assert(abs(p - 2.839521721752) < eps);

    cout << "Test PDF Passed!!\n";
    cout << "----------------------------\n";
}

int main() {

    MatrixTest();
    pdfTest();

    clock_t start = clock();

    int L = 5;
    double outer_sample = 2000;

    theta_u[0] = 1;
    theta_u[1] = 0;

    theta_sigma[0][0] = theta_sigma[1][1] = 2;
    theta_sigma[0][1] = theta_sigma[1][0] = -1;

    A[0][0] = 1;
    A[0][1] = A[1][0] = 2;
    A[1][1] = A[2][0] = 3;
    A[2][1] = 4;

    InvA[0][0] = InvA[1][0] = InvA[1][1] = 1;
    InvA[0][1] = -6; InvA[0][2] = 4; InvA[1][2] = -1;

    epsilon_sigma[0][0] = epsilon_sigma[1][1] = epsilon_sigma[2][2] = 0.1;
    epsilon_sigma[0][1] = epsilon_sigma[1][0] = epsilon_sigma[2][1] = epsilon_sigma[1][2] = -0.05;

    estimate_alpha_beta(L, outer_sample);

    clock_t end = clock();
    cout << "time = " << (double)(end - start) / CLOCKS_PER_SEC << " (sec)\n";

    // M=2, L=5, N=2000 で time = 214.767 (sec)

    return 0;
}