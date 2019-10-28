
#include <random>
#include <iostream>

using namespace std;

#include "../evppi.hpp"

random_device rd;
mt19937 generator(rd());
normal_distribution<double> dist(0.0, 1.0);

struct Matrix {
    vector < vector<double> > val;
    Matrix(int n, int m) { val.clear(); val.resize(n, vector<double>(m)); }
    int size() { return val.size(); }
    inline vector<double> &operator[](int i) { return val[i]; }
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

vector<double> rand_multinormal(vector<double> &u, Matrix &sigma_cholesky) {
    size_t sz = u.size();
    vector<double> rand(sz);
    vector<double> ret(sz);
    for (size_t i = 0; i < sz; i++) {
        rand[i] = dist(generator);
        for (size_t j = 0; j <= i; j++)
            ret[i] += sigma_cholesky[i][j] * rand[j];
        ret[i] += u[i];
    }
    return ret;
}

int n_sim = 20000;
int wtp = 20000;
double ssi_qaly_loss = 0.12;
vector <double> dressing_cost(4);

double ssi_cost_mean = 8.972237608;
double ssi_cost_s = 0.163148238;
lognormal_distribution<double> ssi_cost(ssi_cost_mean, ssi_cost_s);

double p_ssi_mean = 0.137984898;
double p_ssi_se = 0.001772214;
normal_distribution<double> p_ssi_dist(p_ssi_mean, p_ssi_se);

vector <double> u(3);
Matrix Sigma(3, 3);
Matrix sigma_cholesky(3, 3);

double expit(double x) {
    return exp(x) / (1 + exp(x));
}

double logit(double x) {
    return log(x / (1 - x));
}

/*
 * わかりにくくてすまん...構造体をうまく使いたい...
 * sample[0~3]: pSSI
 * sample[4]: SSIcost
 * sample[5~7]: d
 */

void sampling_init(EvppiInfo *info) {
    info->model_num = 4;
    info->sample.resize(8);
    info->val.resize(info->model_num);
}

void pre_sampling(EvppiInfo *info) {
    vector <double> r = rand_multinormal(u, sigma_cholesky);
    info->sample[5] = r[0];
    info->sample[6] = r[1];
    info->sample[7] = r[2];
}

void post_sampling(EvppiInfo *info) {
    info->sample[0] = p_ssi_dist(generator);
    info->sample[1] = expit(logit(info->sample[0]) + info->sample[5]);
    info->sample[2] = expit(logit(info->sample[0]) + info->sample[6]);
    info->sample[3] = expit(logit(info->sample[0]) + info->sample[7]);
    info->sample[4] = ssi_cost(generator);
}

void f(EvppiInfo *info) {
    info->val[0] = -dressing_cost[0] - info->sample[0] * (info->sample[4] + ssi_qaly_loss * wtp);
    info->val[1] = -dressing_cost[1] - info->sample[1] * (info->sample[4] + ssi_qaly_loss * wtp);
    info->val[2] = -dressing_cost[2] - info->sample[2] * (info->sample[4] + ssi_qaly_loss * wtp);
    info->val[3] = -dressing_cost[3] - info->sample[3] * (info->sample[4] + ssi_qaly_loss * wtp);
}

// params for mlmc
int m0 = 1;
int s = 2;
int max_level = 20;
int test_level = 10;

int main() {
    dressing_cost[0] = 0.0;
    dressing_cost[1] = 5.25;
    dressing_cost[2] = 13.86;
    dressing_cost[3] = 21.39;

    u[0] = 0.05021921904305;
    u[1] = -0.06629096195095;
    u[2] = -0.178047360396;

    Sigma[0][0] = 0.06546909;
    Sigma[0][1] = 0.06274766;
    Sigma[0][2] = 0.01781241;
    Sigma[1][0] = 0.06274766;
    Sigma[1][1] = 0.21792037;
    Sigma[1][2] = 0.01727998;
    Sigma[2][0] = 0.01781241;
    Sigma[2][1] = 0.01727998;
    Sigma[2][2] = 0.04631648;

    sigma_cholesky = Cholesky(Sigma);

    MlmcInfo *info = mlmc_init(m0, s, max_level, 1.0, 0.25);
    mlmc_test(info, test_level, n_sim);

    vector <double> eps;
    eps.push_back(0.02);
    eps.push_back(0.01);

    mlmc_test_eval_eps(info, eps);

    return 0;
}
