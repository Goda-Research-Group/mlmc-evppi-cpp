
#include <random>
#include <iostream>

using namespace std;

#include "../evppi.hpp"
#include "../matrix.hpp"
#include "../util.hpp"

random_device rd;
mt19937 generator(rd());

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
 * sample[0]: pSSI2
 * sample[1]: SSIcost
 * sample[2~4]: d
 */

void sampling_init(EvppiInfo *info) {
    info->model_num = 4;
    info->sample.resize(5);
    info->val.resize(info->model_num);
}

void pre_sampling(EvppiInfo *info) {
    vector <double> rand(3), ret(3);
    rand[0] = dist_normal(generator);
    rand[1] = dist_normal(generator);
    rand[2] = dist_normal(generator);
    rand_multinormal(u, sigma, rand, ret);

    info->sample[2] = ret[0];
    info->sample[3] = ret[1];
    info->sample[4] = ret[2];
}

void post_sampling(EvppiInfo *info) {
    info->sample[0] = ssi_risk(generator);
    info->sample[1] = ssi_cost(generator);
}

void f(EvppiInfo *info) {
    int wtp = 20000;
    double ssi_qaly_loss = 0.12;
    vector <double> dressing_cost = {0.0, 5.25, 13.86, 21.39};

    double pSSI1 = expit(logit(info->sample[0]) + info->sample[2]);
    double pSSI2 = info->sample[0];
    double pSSI3 = expit(logit(info->sample[0]) + info->sample[3]);
    double pSSI4 = expit(logit(info->sample[0]) + info->sample[4]);

    info->val[0] = -dressing_cost[0] - pSSI1 * (info->sample[1] + ssi_qaly_loss * wtp);
    info->val[1] = -dressing_cost[1] - pSSI2 * (info->sample[1] + ssi_qaly_loss * wtp);
    info->val[2] = -dressing_cost[2] - pSSI3 * (info->sample[1] + ssi_qaly_loss * wtp);
    info->val[3] = -dressing_cost[3] - pSSI4 * (info->sample[1] + ssi_qaly_loss * wtp);
}

int main() {
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

    MlmcInfo *info = mlmc_init(1, 2, 20, 1.0, 0.25);
    smc_evpi_calc(info->layer[0].evppi_info, 10000000);
    mlmc_test(info, 10, 200000);

    vector <double> eps = {0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001};
    mlmc_test_eval_eps(info, eps);

    return 0;
}
