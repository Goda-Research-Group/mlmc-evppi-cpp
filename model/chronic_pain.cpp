
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

gamma_distribution<double> cost_med_t1(100, 0.0226);       // param of 2.1*Inflation
gamma_distribution<double> cost_med_t2(100, 0.00030137);   // param of 0.04*Inflation*(1-0.3)
gamma_distribution<double> cost_ae(100, 0.06991);          // param of 6.991009409
gamma_distribution<double> cost_withdraw_ae(100, 1.0691); // param of 106.911031273
gamma_distribution<double> cost_withdraw(100, 1.0691);    // param of 106.911031273
gamma_distribution<double> cost_discontinue(100, 0.185);   // param of 18.5

/*
 ** first round
 * info->sample[0]:  probability of adverse effects,                   treatment 1
 * info->sample[1]:  probability of withdrawal due to adverse effects, treatment 1
 * info->sample[2]:  probability of withdrawal due to adverse effects, treatment 2
 * info->sample[3]:  probability of withdrawal due to other reasons,   treatment 1
 * info->sample[4]:  probability of withdrawal due to other reasons,   treatment 2
 * info->sample[5]:  probability of discontinuation
 *
 ** second round
 * info->sample[6]:  probability of adverse effects,                   treatment 1
 * info->sample[7]:  probability of adverse effects,                   treatment 2
 * info->sample[8]:  probability of withdrawal due to adverse effects, treatment 1
 * info->sample[9]:  probability of withdrawal due to adverse effects, treatment 2
 * info->sample[10]: probability of withdrawal due to other reasons
 * info->sample[11]: probability of discontinuation (not used)
 *
 ** cost
 * info->sample[12]: cost of treatment 1
 * info->sample[13]: cost of treatment 2
 * info->sample[14]: cost of adverse events
 * info->sample[15]: cost of withdrawing due to adverse effects
 * info->sample[16]: cost of withdrawing due to other reasons
 * info->sample[17]: cost of discontinuing
 *
 ** utility
 * info->sample[18]: no adverse effects
 * info->sample[19]: adverse effects
 * info->sample[20]: withdraw from treatment due to adverse effects
 * info->sample[21]: withdraw due to other reasons
 */

void sampling_init(EvppiInfo *info) {
    info->model_num = 2;
    info->sample.resize(22);
    info->val.resize(info->model_num);
}

void pre_sampling(EvppiInfo *info) {
    info->sample[12] = cost_med_t1(generator);
    info->sample[13] = cost_med_t2(generator);
    info->sample[14] = cost_ae(generator);
    info->sample[15] = cost_withdraw_ae(generator);
    info->sample[16] = cost_withdraw_ae(generator);
    info->sample[17] = cost_discontinue(generator);
    info->sample[18] = beta(29.805, 13.0799); // param of 0.695000000
    info->sample[19] = beta(41.117, 29.4096); // param of 0.583000000
    info->sample[20] = beta(49.197, 48.6102); // param of 0.503000000
    info->sample[21] = beta(59.095, 86.8186); // param of 0.405000000
}

void post_sampling(EvppiInfo *info) {
    info->sample[0]  = beta(55.9479, 72.3261); // param of 0.436159243
    info->sample[1]  = beta(94.3703, 1598.69); // param of 0.055739588
    info->sample[2]  = beta(97.6812, 4157.01); // param of 0.022958454
    info->sample[3]  = beta(98.7131, 7648.68); // param of 0.012741455
    info->sample[4]  = beta(99.8371, 61818.2); // param of 0.001612408
    info->sample[5]  = beta(94.9500, 1804.05); // param of 0.050000000
    info->sample[6]  = beta(53.1865, 61.5632); // param of 0.463500000
    info->sample[7]  = beta(53.1865, 61.5632); // param of 0.463500000
    info->sample[8]  = beta(96.6874, 2851.30); // param of 0.032797792
    info->sample[9]  = beta(96.6874, 2851.30); // param of 0.032797792
    info->sample[10] = beta(99.7674, 43212.6); // param of 0.002303439
    info->sample[11] = beta(89.9000, 809.100); // param of 0.100000000
}

void f(EvppiInfo *info) {
    vector <Matrix> transition(2, Matrix(10, 10));

    // treatment 1
    transition[0][0][0] = (1 - info->sample[1] - info->sample[3]) * (1 - info->sample[0]);
    transition[0][0][1] = (1 - info->sample[1] - info->sample[3]) * info->sample[0];
    transition[0][0][2] = info->sample[1];
    transition[0][0][3] = info->sample[3];
    transition[0][1][0] = transition[0][0][0];
    transition[0][1][1] = transition[0][0][1];
    transition[0][1][2] = transition[0][0][2];
    transition[0][1][3] = transition[0][0][3];
    transition[0][2][4] = (1 - info->sample[5]) * (1 - info->sample[6]);
    transition[0][2][5] = (1 - info->sample[5]) * info->sample[6];
    transition[0][2][9] = info->sample[5];
    transition[0][3][4] = transition[0][2][4];
    transition[0][3][5] = transition[0][2][5];
    transition[0][3][9] = transition[0][2][9];
    transition[0][4][4] = (1 - info->sample[8] - info->sample[10]) * (1 - info->sample[6]);
    transition[0][4][5] = (1 - info->sample[8] - info->sample[10]) * info->sample[6];
    transition[0][4][6] = info->sample[8];
    transition[0][4][7] = info->sample[10];
    transition[0][5][4] = transition[0][4][4];
    transition[0][5][5] = transition[0][4][5];
    transition[0][5][6] = transition[0][4][6];
    transition[0][5][7] = transition[0][4][7];
    transition[0][6][9] = 1;
    transition[0][7][9] = 1;
    transition[0][8][8] = 1;
    transition[0][9][9] = 1;

    // treatment 2
    transition[1][0][0] = (1 - info->sample[2] - info->sample[4]) * (1 - info->sample[0] * 0.7);
    transition[1][0][1] = (1 - info->sample[2] - info->sample[4]) * info->sample[0] * 0.7;
    transition[1][0][2] = info->sample[2];
    transition[1][0][3] = info->sample[4];
    transition[1][1][0] = transition[1][0][0];
    transition[1][1][1] = transition[1][0][1];
    transition[1][1][2] = transition[1][0][2];
    transition[1][1][3] = transition[1][0][3];
    transition[1][2][4] = (1 - info->sample[5]) * (1 - info->sample[7]);
    transition[1][2][5] = (1 - info->sample[5]) * info->sample[7];
    transition[1][2][9] = info->sample[5];
    transition[1][3][4] = transition[1][2][4];
    transition[1][3][5] = transition[1][2][5];
    transition[1][3][9] = transition[1][2][9];
    transition[1][4][4] = (1 - info->sample[9] - info->sample[10]) * (1 - info->sample[7]);
    transition[1][4][5] = (1 - info->sample[9] - info->sample[10]) * info->sample[7];
    transition[1][4][6] = info->sample[9];
    transition[1][4][7] = info->sample[10];
    transition[1][5][4] = transition[1][4][4];
    transition[1][5][5] = transition[1][4][5];
    transition[1][5][6] = transition[1][4][6];
    transition[1][5][7] = transition[1][4][7];
    transition[1][6][9] = 1;
    transition[1][7][9] = 1;
    transition[1][8][8] = 1;
    transition[1][9][9] = 1;

    // cost
    vector < vector <double> > cost(2, vector<double>(10));
    cost[0][0] = 2.6327 + info->sample[12];
    cost[1][0] = 55.207 + info->sample[13];
    cost[0][1] = cost[0][0] + info->sample[14];
    cost[1][1] = cost[1][0] + info->sample[14];
    cost[0][2] = cost[1][2] = info->sample[15];
    cost[0][3] = cost[1][3] = info->sample[16];
    cost[0][4] = cost[1][4] = 9.2442;
    cost[0][5] = cost[1][5] = 9.2442 + info->sample[14];
    cost[0][6] = cost[1][6] = info->sample[15];
    cost[0][7] = cost[1][7] = info->sample[16];
    cost[0][8] = cost[1][8] = 4.893;
    cost[0][9] = cost[1][9] = info->sample[17];

    // util
    vector <double> util(10);
    util[0] = info->sample[18] * 0.019165;
    util[1] = info->sample[19] * 0.019165;
    util[2] = info->sample[20] * 0.019165;
    util[3] = info->sample[21] * 0.019165;
    util[4] = info->sample[18] * 0.017248;
    util[5] = info->sample[19] * 0.017248;
    util[6] = info->sample[20] * 0.017248;
    util[7] = info->sample[21] * 0.017248;
    util[8] = (info->sample[18] + info->sample[19]) * 0.0095825;
    util[9] = info->sample[21] * 0.015332;

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
    // smc_evpi_calc(info->layer[0].evppi_info, 1000000);
    mlmc_test(info, 10, 2000);

    vector <double> eps = {5, 2, 1, 0.5, 0.2, 0.1};
    mlmc_test_eval_eps(info, eps);

    return 0;
}
