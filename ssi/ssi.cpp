
#include <random>
#include <iostream>

using namespace std;

#include "../evppi.hpp"
#include "../matrix.hpp"
#include "../util.hpp"

random_device rd;
mt19937 generator(rd());

normal_distribution<double> dist_normal(0.0, 1.0);
lognormal_distribution<double> ssi_cost(8.97224, 0.163148);
normal_distribution<double> ssi_risk(0.137985, 0.00177221); // basecase
// normal_distribution<double> ssi_risk(0.0893905, 0.00606212); // Jenks

struct ModelInfo {
    double mu;
    double d1, d3, d4;
    double SSI_cost;
};

void sampling_init(EvppiInfo *info) {
    info->model_num = 4;
    info->model_info = new ModelInfo;
    info->val.resize(info->model_num);
}

void pre_sampling(ModelInfo *model) {
    vector <double> mu = {-0.0502192, -0.0662909, -0.178047};
    Matrix triangular_matrix(3, 3);
    triangular_matrix[0] = {0.255869, 0, 0};
    triangular_matrix[1] = {0.245233, 0.397217, 0};
    triangular_matrix[2] = {0.0696153, 0.000523642, 0.203642};
    vector <double> rand(3), ret(3);
    rand[0] = dist_normal(generator);
    rand[1] = dist_normal(generator);
    rand[2] = dist_normal(generator);
    rand_multinormal(mu, triangular_matrix, rand, ret);

    model->d1 = ret[0];
    model->d3 = ret[1];
    model->d4 = ret[2];
}

void post_sampling(ModelInfo *model) {
    model->mu = logit(ssi_risk(generator));
    model->SSI_cost = ssi_cost(generator);
}

void f(EvppiInfo *info) {
    int WTP = 20000;
    double SSI_QALY_loss = 0.12;
    vector <double> dressing_cost = {0.0, 5.25, 13.86, 21.39};

    ModelInfo *model = info->model_info;

    double pSSI1 = expit(model->mu + model->d1);
    double pSSI2 = expit(model->mu);
    double pSSI3 = expit(model->mu + model->d3);
    double pSSI4 = expit(model->mu + model->d4);

    info->val[0] = - dressing_cost[0] - pSSI1 * (model->SSI_cost + SSI_QALY_loss * WTP);
    info->val[1] = - dressing_cost[1] - pSSI2 * (model->SSI_cost + SSI_QALY_loss * WTP);
    info->val[2] = - dressing_cost[2] - pSSI3 * (model->SSI_cost + SSI_QALY_loss * WTP);
    info->val[3] = - dressing_cost[3] - pSSI4 * (model->SSI_cost + SSI_QALY_loss * WTP);
}

int main() {
    MlmcInfo *info = mlmc_init(1, 2, 30, 1.0, 0.25);
    // smc_evpi_calc(info->layer[0].evppi_info, 10000000);
    mlmc_test(info, 10, 200000);

    vector <double> eps = {0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001};
    mlmc_test_eval_eps(info, eps);

    return 0;
}
