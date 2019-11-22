
#include <iostream>
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

static double pi, rho;
vector<double> qaly_of_states(4);
vector<double> cost_of_states(4);
Matrix transition(4, 4);
vector <double> cost_of_drug = {110, 420};

double beta(double alpha, double beta) {
    gamma_distribution<double> dist_gamma1(alpha, 1.0);
    gamma_distribution<double> dist_gamma2(beta, 1.0);
    double r1 = dist_gamma1(generator);
    double r2 = dist_gamma2(generator);
    return r1 / (r1 + r2);
}

/*
 * info->sample[0]  : probability of adverse events
 * info->sample[1]  : reduction in adverse events with treatment
 * info->sample[2]  : QoL weight with no adverse events
 * info->sample[3]  : QoL weight for home care (ambulatory)
 * info->sample[4]  : QoL weight for hospitalization
 * info->sample[5]  : cost of home care (ambulatory)
 * info->sample[6]  : cost of hospitalization
 * info->sample[7]  : cost of death
 * info->sample[8]  : probability of hospitalization
 * info->sample[9]  : daily probability of recovery from home care (ambulatory)
 * info->sample[10] : probability of death
 * info->sample[11] : daily probability of recovery from hospital
 */

void sampling_init(EvppiInfo *info) {
    info->model_num = 2;
    info->sample.resize(12);
    info->val.resize(info->model_num);
}

void pre_sampling(EvppiInfo *info) {
}

void post_sampling(EvppiInfo *info) {
    info->sample[0] = beta(1 + 27, 1 + 111 - 27);
    info->sample[1] = dist_rho(generator);
    info->sample[2] = beta(5.75, 5.75);
    info->sample[3] = beta(0.87, 3.47);
    info->sample[4] = beta(18.23, 0.372);
    info->sample[5] = dist_cost_ambulatory(generator);
    info->sample[6] = dist_cost_hospital(generator);
    info->sample[7] = dist_cost_death(generator);
    info->sample[8] = beta(1 + 17, 1 + 27 - 17);
    info->sample[9] = beta(5.12, 6.26);
    info->sample[10] = beta(1 + 1, 1 + 17 - 1);
    info->sample[11] = beta(3.63, 6.74);
}

void f(EvppiInfo *info) {
    int threshold = 15;
    int wtp = 30000;

    pi = info->sample[0];
    rho = info->sample[1];

    qaly_of_states[0] = info->sample[2];
    qaly_of_states[1] = info->sample[3];
    qaly_of_states[2] = info->sample[4];

    cost_of_states[0] = info->sample[5];
    cost_of_states[1] = info->sample[6];
    cost_of_states[3] = info->sample[7];

    transition[0][1] = info->sample[8] / threshold;                 // ambulatory -> hospital
    transition[0][2] = info->sample[9] * (1.0 - transition[0][1]);  // ambulatory -> recovery
    transition[0][0] = 1.0 - transition[0][1] - transition[0][2];   // stay ambulatory

    transition[1][3] = info->sample[10] / threshold;                // hospital -> death
    transition[1][2] = info->sample[11] * (1.0 - transition[1][3]); // hospital -> recovery
    transition[1][1] = 1.0 - transition[1][2] - transition[1][3];   // stay hospital

    vector <vector<double>> init(2);
    init[0] = {pi, 0.0, 0.0, 0.0};
    init[1] = {pi * rho, 0.0, 0.0, 0.0};

    vector<double> sum_of_effect(2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < threshold; j++) {
            init[i] = init[i] * transition;
            sum_of_effect[i] += qaly_of_states * init[i];
        }
        sum_of_effect[i] /= threshold + 1;
    }

    init[0] = {pi, 0.0, 0.0, 0.0};
    init[1] = {pi * rho, 0.0, 0.0, 0.0};

    vector<double> sum_of_cost(2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < threshold; j++) {
            init[i] = init[i] * transition;
            sum_of_cost[i] += cost_of_states * init[i];
        }
        sum_of_cost[i] /= threshold + 1;
    }

    for (int i = 0; i < 2; i++) {
        info->val[i] =
                (sum_of_effect[0] + (1.0 - pi * (i ? rho : 1)) * qaly_of_states[2]) * wtp
                - (sum_of_cost[i] + cost_of_drug[i]);
    }
}

int main() {
    qaly_of_states[3] = 0.0;
    cost_of_states[2] = 0.0;

    transition[0][3] = 0.0;
    transition[1][0] = 0.0;
    transition[2] = {0.0, 0.0, 1.0, 0.0};
    transition[3] = {0.0, 0.0, 0.0, 1.0};

    MlmcInfo *info = mlmc_init(1, 2, 20, 1.0, 0.25);
    smc_evpi_calc(info->layer[0].evppi_info, 100000);
    mlmc_test(info, 10, 2000);

    return 0;
}