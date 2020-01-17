
#include <random>

#include "../matrix.hpp"
#include "../evppi.hpp"

using namespace std;

random_device rd;
mt19937 generator(rd());

double beta(double alpha, double beta) {
    gamma_distribution<double> dist_gamma1(alpha, 1.0);
    gamma_distribution<double> dist_gamma2(beta, 1.0);
    double r1 = dist_gamma1(generator);
    double r2 = dist_gamma2(generator);
    return r1 / (r1 + r2);
}

gamma_distribution<double> cost_morphine(100, 0.0226);      // param of 2.1 * Inflation
gamma_distribution<double> cost_oxycodone(100, 0.00030137); // param of 0.04 * Inflation
gamma_distribution<double> cost_ae(100, 0.06991);           // param of 6.991
gamma_distribution<double> cost_withdrawal(100, 1.0691);    // param of 106.91
gamma_distribution<double> cost_discontinue(100, 0.185);    // param of 18.50

/*
 ** Probability
 * info->sample[0] = 0.436 // probability of AE, morphine
 * info->sample[1] = 0.056 // probability of withdrawal due to AE, morphine
 * info->sample[2] = 0.013 // probability of withdrawal due to OR, morphine
 * info->sample[3] = 0.464 // probability of AE, oxycodone
 * info->sample[4] = 0.033 // probability of withdrawal due to AE, oxycodone
 * info->sample[5] = 0.002 // probability of withdrawal due to OR, oxycodone
 * info->sample[6] = 0.050 // probability of discontinuation
 *
 ** Cost
 * info->sample[7]  = 2.26   // co-medication cost of morphine
 * info->sample[8]  = 0.043  // co-medication cost of oxycodone
 * info->sample[9]  = 6.99   // cost of AE
 * info->sample[10] = 106.91 // cost of withdrawal
 * info->sample[11] = 18.50  // cost of discontinuation
 *
 ** Utility
 * info->sample[12] = 0.695 // utility of noAE
 * info->sample[13] = 0.583 // utility of AE
 * info->sample[14] = 0.503 // withdrawal due to AE
 * info->sample[15] = 0.405 // withdrawal due to OR
 */

void sampling_init(EvppiInfo *info) {
    info->model_num = 2;
    info->sample.resize(16);
    info->val.resize(info->model_num);
}

void pre_sampling(EvppiInfo *info) {
    info->sample[7] = cost_morphine(generator);
    info->sample[8] = cost_oxycodone(generator);
    info->sample[9] = cost_ae(generator);
    info->sample[10] = cost_withdrawal(generator);
    info->sample[11] = cost_discontinue(generator);

    info->sample[12] = beta(29.805, 13.0799) * 7 / 365.25; // param of 0.695
    info->sample[13] = beta(41.117, 29.4096) * 7 / 365.25; // param of 0.583
    info->sample[14] = beta(49.197, 48.6102) * 7 / 365.25; // param of 0.503
    info->sample[15] = beta(59.095, 86.8186) * 7 / 365.25; // param of 0.405
}

void post_sampling(EvppiInfo *info) {
    info->sample[0]  = beta(55.9479, 72.3261); // param of 0.436
    info->sample[1]  = beta(94.3703, 1598.69); // param of 0.056
    info->sample[2]  = beta(98.7131, 7648.68); // param of 0.013
    info->sample[3]  = beta(53.1865, 61.5632); // param of 0.464
    info->sample[4]  = beta(96.6874, 2851.30); // param of 0.033
    info->sample[5]  = beta(99.8371, 61818.2); // param of 0.002
    info->sample[6]  = beta(94.9500, 1804.05); // param of 0.050
}

void f(EvppiInfo *info) {
    vector <Matrix> transition(2, Matrix(10, 10));

    transition[0][0][0] = (1 - info->sample[1] - info->sample[2]) * (1 - info->sample[0]);
    transition[0][0][1] = (1 - info->sample[1] - info->sample[2]) * info->sample[0];
    transition[0][0][2] = info->sample[1];
    transition[0][0][3] = info->sample[2];
    transition[0][1][0] = transition[0][0][0];
    transition[0][1][1] = transition[0][0][1];
    transition[0][1][2] = transition[0][0][2];
    transition[0][1][3] = transition[0][0][3];
    transition[0][2][4] = (1 - info->sample[6]) * (1 - info->sample[3]);
    transition[0][2][5] = (1 - info->sample[6]) * info->sample[3];
    transition[0][2][9] = info->sample[6];
    transition[0][3][4] = transition[0][2][4];
    transition[0][3][5] = transition[0][2][5];
    transition[0][3][9] = transition[0][2][9];
    transition[0][4][4] = (1 - info->sample[4] - info->sample[5]) * (1 - info->sample[3]);
    transition[0][4][5] = (1 - info->sample[4] - info->sample[5]) * info->sample[3];
    transition[0][4][6] = info->sample[4];
    transition[0][4][7] = info->sample[5];
    transition[0][5][4] = transition[0][4][4];
    transition[0][5][5] = transition[0][4][5];
    transition[0][5][6] = transition[0][4][6];
    transition[0][5][7] = transition[0][4][7];
    transition[0][6][9] = 1;
    transition[0][7][9] = 1;
    transition[0][8][8] = 1;
    transition[0][9][9] = 1;

    transition[1][0][0] = (1 - info->sample[4] * 0.7 - info->sample[5] * 0.7) * (1 - info->sample[3] * 0.7);
    transition[1][0][1] = (1 - info->sample[4] * 0.7 - info->sample[5] * 0.7) * info->sample[3] * 0.7;
    transition[1][0][2] = info->sample[4] * 0.7;
    transition[1][0][3] = info->sample[5] * 0.7;
    transition[1][1][0] = transition[1][0][0];
    transition[1][1][1] = transition[1][0][1];
    transition[1][1][2] = transition[1][0][2];
    transition[1][1][3] = transition[1][0][3];
    transition[1][2][4] = (1 - info->sample[6]) * (1 - info->sample[3]);
    transition[1][2][5] = (1 - info->sample[6]) * info->sample[3];
    transition[1][2][9] = info->sample[6];
    transition[1][3][4] = transition[0][2][4];
    transition[1][3][5] = transition[0][2][5];
    transition[1][3][9] = transition[0][2][9];
    transition[1][4][4] = (1 - info->sample[4] - info->sample[5]) * (1 - info->sample[3]);
    transition[1][4][5] = (1 - info->sample[4] - info->sample[5]) * info->sample[3];
    transition[1][4][6] = info->sample[4];
    transition[1][4][7] = info->sample[5];
    transition[1][5][4] = transition[0][4][4];
    transition[1][5][5] = transition[0][4][5];
    transition[1][5][6] = transition[0][4][6];
    transition[1][5][7] = transition[0][4][7];
    transition[1][6][9] = 1;
    transition[1][7][9] = 1;
    transition[1][8][8] = 1;
    transition[1][9][9] = 1;

    vector < vector <double> > cost(2, vector<double>(10));
    cost[0][0] = 2.63 + info->sample[7];
    cost[1][0] = 55.21 + info->sample[8] * 0.7;
    cost[0][1] = cost[0][0] + info->sample[9];
    cost[1][1] = cost[1][0] + info->sample[9];
    cost[0][2] = cost[1][2] = info->sample[10];
    cost[0][3] = cost[1][3] = info->sample[10];
    cost[0][4] = cost[1][4] = 9.20 + info->sample[8];
    cost[0][5] = cost[1][5] = 9.20 + info->sample[8] + info->sample[9];
    cost[0][6] = cost[1][6] = info->sample[10];
    cost[0][7] = cost[1][7] = info->sample[10];
    cost[0][8] = cost[1][8] = 4.893;
    cost[0][9] = cost[1][9] = info->sample[11];

    // util
    vector <double> util(10);
    util[0] = info->sample[12];
    util[1] = info->sample[13];
    util[2] = info->sample[14];
    util[3] = info->sample[15];
    util[4] = info->sample[12] * 0.9;
    util[5] = info->sample[13] * 0.9;
    util[6] = info->sample[14] * 0.9;
    util[7] = info->sample[15] * 0.9;
    util[8] = (info->sample[12] + info->sample[13]) / 2;
    util[9] = info->sample[15] * 0.8;

    for (int i = 0; i < 2; i++) {
        double sum_of_cost = 0, sum_of_effect = 0;
        vector <double> init(10);
        init[0] = 1;

        for (int t = 0; t < 51; t++) {
            vector<double> now = init * transition[i];
            vector<double> half_cycle_corrected(10);
            for (int j = 0; j < 10; j++) {
                half_cycle_corrected[j] = (init[j] + now[j]) / 2;
            }

            init = now;
            sum_of_cost += cost[i] * half_cycle_corrected;
            sum_of_effect += util * half_cycle_corrected;
        }

        sum_of_cost += cost[i] * init;
        sum_of_effect += util * init;

        info->val[i] = 12.5174 * (sum_of_effect * 20000 - sum_of_cost);
    }
}

int main() {

    MlmcInfo *info = mlmc_init(1, 2, 20, 1.0, 0.25);
    // smc_evpi_calc(info->layer[0].evppi_info, 1000000); // evpi = 1085
    mlmc_test(info, 10, 2000);

    vector <double> eps = {5, 2, 1, 0.5, 0.2, 0.1};
    mlmc_test_eval_eps(info, eps);

    return 0;
}
