
#include <iostream>
#include <iomanip>
#include <random>

#include "../util.hpp"
#include "../matrix.hpp"
#include "../evppi.hpp"

using namespace std;

random_device rd;
mt19937 generator(rd());

normal_distribution<double> dist_rho(0.65, 0.1);
lognormal_distribution<double> dist_cost_ambulatory(7.74, 0.039);
lognormal_distribution<double> dist_cost_hospital(8.77, 0.15);
lognormal_distribution<double> dist_cost_death(8.33, 0.13);

double beta(double alpha, double beta) {
    gamma_distribution<double> dist_gamma1(alpha, 1.0);
    gamma_distribution<double> dist_gamma2(beta, 1.0);
    double r1 = dist_gamma1(generator);
    double r2 = dist_gamma2(generator);
    return r1 / (r1 + r2);
}

struct ModelInfo {
    double pi0, rho;
    double q1, q2, q3;
    double c1, c2, c4;
    double gamma1, gamma2;
    double lambda1, lambda2;
};

void sampling_init(EvppiInfo *info) {
    info->model_num = 2;
    info->model_info = new ModelInfo;
    info->val.resize(info->model_num);
}

void pre_sampling(ModelInfo *model) {
    model->pi0 = beta(28, 85);
    model->rho = dist_rho(generator);
    model->gamma1 = beta(11, 18) / 15;
    model->gamma2 = beta(2, 17) / 15;
    model->lambda1 = beta(5.12, 6.26);
    model->lambda2 = beta(3.63, 6.74);
}

void post_sampling(ModelInfo *model) {
    model->q1 = beta(5.75, 5.75);
    model->q2 = beta(0.87, 3.47);
    model->q3 = beta(18.23, 0.372);
    model->c1 = dist_cost_ambulatory(generator);
    model->c2 = dist_cost_hospital(generator);
    model->c4 = dist_cost_death(generator);
}

void f(EvppiInfo *info) {
    ModelInfo *model = info->model_info;

    vector <double> qaly = {model->q1, model->q2, model->q3, 0};
    vector <double> cost = {model->c1, model->c2, 0, model->c4};

    Matrix transition(4, 4);

    transition[0][0] = (1 - model->lambda1) * (1 - model->gamma1);
    transition[0][1] = model->gamma1;
    transition[0][2] = model->lambda1 * (1 - model->gamma1);
    transition[0][3] = 0;

    transition[1][0] = 0;
    transition[1][1] = (1 - model->lambda2) * (1 - model->gamma2);
    transition[1][2] = model->lambda2 * (1 - model->gamma2);
    transition[1][3] = model->gamma2;

    transition[2] = {0, 0, 1, 0};
    transition[3] = {0, 0, 0, 1};

    vector <vector<double>> init(2);
    init[0] = {model->pi0, 0.0, 0.0, 0.0};
    init[1] = {model->pi0 * model->rho, 0.0, 0.0, 0.0};

    vector<double> sum_of_qaly(2), sum_of_cost(2);;
    for (int i = 0; i < 2; i++) {
        sum_of_qaly[i] += qaly * init[i];
        sum_of_cost[i] += cost * init[i];

        for (int j = 0; j < 15; j++) {
            init[i] = init[i] * transition;
            sum_of_qaly[i] += qaly * init[i];
            sum_of_cost[i] += cost * init[i];
        }

        sum_of_qaly[i] /= 16;
        sum_of_cost[i] /= 16;
    }

    sum_of_qaly[0] += (1 - model->pi0) * model->q3;
    sum_of_qaly[1] += (1 - model->pi0 * model->rho) * model->q3;

    sum_of_cost[0] += 110;
    sum_of_cost[1] += 420;

    int wtp = 30000;
    for (int i = 0; i < 2; i++) {
        info->val[i] = sum_of_qaly[i] * wtp - sum_of_cost[i];
    }
}

int main() {
    MlmcInfo *info = mlmc_init(1, 2, 20, 1.0, 0.25);
    smc_evpi_calc(info->layer[0].evppi_info, 1000000); // 33.09
    mlmc_test(info, 10, 20000);

    vector <double> eps = {0.2, 0.1, 0.05, 0.02, 0.01};
    mlmc_test_eval_eps(info, eps);

    return 0;
}