
#include <random>
#include <iostream>

using namespace std;

#include "../evppi.hpp"
#include "../matrix.hpp"
#include "../util.hpp"

random_device rd;
mt19937 generator(rd());

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
    info->sample[4] = ssi_cost(generator);
}

void f(EvppiInfo *info) {
    info->sample[1] = expit(logit(info->sample[0]) + info->sample[5]);
    info->sample[2] = expit(logit(info->sample[0]) + info->sample[6]);
    info->sample[3] = expit(logit(info->sample[0]) + info->sample[7]);

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
    dressing_cost = {0.0, 5.25, 13.86, 21.39};
    u = {0.05021921904305, -0.06629096195095, -0.178047360396};
    Sigma[0] = {0.06546909, 0.06274766, 0.01781241};
    Sigma[1] = {0.06274766, 0.21792037, 0.01727998};
    Sigma[2] = {0.01781241, 0.01727998, 0.04631648};
    sigma_cholesky = Cholesky(Sigma);

    MlmcInfo *info = mlmc_init(m0, s, max_level, 1.0, 0.25);
    mlmc_test(info, test_level, n_sim);

    vector <double> eps = {0.0002, 0.0001, 0.00005, 0.00001, 0.00002};
    mlmc_test_eval_eps(info, eps);

    return 0;
}
