
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

normal_distribution<double> dist_normal(0.0, 1.0);

double ssi_cost_mean = 8.972237608;
double ssi_cost_s = 0.163148238;
lognormal_distribution<double> ssi_cost(ssi_cost_mean, ssi_cost_s);

double ssi_risk_mean = 0.137984898; // basecase:0.137984898, Jenks:0.089390519
double ssi_risk_s = 0.001772214; // basecase:0.001772214, Jenks:0.006062126
normal_distribution<double> ssi_risk(ssi_risk_mean, ssi_risk_s);

vector <double> u(3);
Matrix sigma(3, 3);

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
    vector <double> rand(3), ret(3);
    rand[0] = dist_normal(generator);
    rand[1] = dist_normal(generator);
    rand[2] = dist_normal(generator);
    rand_multinormal(u, sigma, rand, ret);

    info->sample[5] = ret[0];
    info->sample[6] = ret[1];
    info->sample[7] = ret[2];
}

void post_sampling(EvppiInfo *info) {
    info->sample[1] = ssi_risk(generator);
    info->sample[4] = ssi_cost(generator);
}

void f(EvppiInfo *info) {
    info->sample[0] = expit(logit(info->sample[1]) + info->sample[5]);
    info->sample[2] = expit(logit(info->sample[1]) + info->sample[6]);
    info->sample[3] = expit(logit(info->sample[1]) + info->sample[7]);

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

    // NMA_all.txt
    u = {-0.05021921904305, -0.06629096195095, -0.178047360396};
    sigma[0] = {0.06546909, 0.06274766, 0.01781241};
    sigma[1] = {0.06274766, 0.21792037, 0.01727998};
    sigma[2] = {0.01781241, 0.01727998, 0.04631648};

    // NMA_BBpop.txt
//    u = {-0.1541009677788, -0.15633834317130002, -0.3795733203173};
//    sigma[0] = {0.08715855, 0.086016,   0.05377359};
//    sigma[1] = {0.086016,   0.24497493, 0.05284933};
//    sigma[2] = {0.05377359, 0.05284933, 0.13153769};

    sigma = Cholesky(sigma);

    MlmcInfo *info = mlmc_init(m0, s, max_level, 1.0, 0.25);
    mlmc_test(info, test_level, n_sim);

    // smc_evpi_calc(info->layer[0].evppi_info, 10000000);

    vector <double> eps = {0.0002, 0.0001, 0.00005, 0.00001, 0.00002};
    mlmc_test_eval_eps(info, eps);

    return 0;
}
