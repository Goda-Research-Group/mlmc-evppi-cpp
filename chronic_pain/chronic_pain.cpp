
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

struct ModelInfo {
    double P_AE_morphine, P_AE_oxycodone;                       // probability of AE
    double P_withdrawal_AE_morphine, P_withdrawal_AE_oxycodone; // probability of withdrawal due to AE
    double P_withdrawal_OR_morphine, P_withdrawal_OR_oxycodone; // probability of withdrawal due to other reasons
    double P_discontinue; // probability of discontinuation

    double C_morphine, C_oxycodone; // co-medication cost
    double C_AE;                    // cost of AE
    double C_withdrawal;            // cost of withdrawal
    double C_discontinue;           // cost of discontinuation

    double U_no_AE;         // utility of no AE
    double U_AE;            // utility of AE
    double U_withdrawal_AE; // utility of withdrawal due to AE
    double U_withdrawal_OR; // utility of withdrawal due to other reasons
};

void sampling_init(EvppiInfo *info) {
    info->model_num = 2;
    info->model_info = new ModelInfo;
    info->val.resize(info->model_num);
}

void pre_sampling(ModelInfo *model) {
    model->C_morphine    = cost_morphine(generator);
    model->C_oxycodone   = cost_oxycodone(generator);
    model->C_AE          = cost_ae(generator);
    model->C_withdrawal  = cost_withdrawal(generator);
    model->C_discontinue = cost_discontinue(generator);

    model->U_no_AE         = beta(29.805, 13.0799) * 7 / 365.25;
    model->U_AE            = beta(41.117, 29.4096) * 7 / 365.25;
    model->U_withdrawal_AE = beta(49.197, 48.6102) * 7 / 365.25;
    model->U_withdrawal_OR = beta(59.095, 86.8186) * 7 / 365.25;
}

void post_sampling(ModelInfo *model) {
    model->P_AE_morphine             = beta(55.9479, 72.3261);
    model->P_withdrawal_AE_morphine  = beta(94.3703, 1598.69);
    model->P_withdrawal_OR_morphine  = beta(98.7131, 7648.68);
    model->P_AE_oxycodone            = beta(53.1865, 61.5632);
    model->P_withdrawal_AE_oxycodone = beta(96.6874, 2851.30);
    model->P_withdrawal_OR_oxycodone = beta(99.8371, 61818.2);
    model->P_discontinue             = beta(94.9500, 1804.05);
}

void f(EvppiInfo *info) {
    ModelInfo *model = info->model_info;

    vector <Matrix> transition(2, Matrix(10, 10));

    transition[0][0][0] = (1 - model->P_withdrawal_AE_morphine - model->P_withdrawal_OR_morphine) * (1 - model->P_AE_morphine);
    transition[0][0][1] = (1 - model->P_withdrawal_AE_morphine - model->P_withdrawal_OR_morphine) * model->P_AE_morphine;
    transition[0][0][2] = model->P_withdrawal_AE_morphine;
    transition[0][0][3] = model->P_withdrawal_OR_morphine;
    transition[0][1][0] = transition[0][0][0];
    transition[0][1][1] = transition[0][0][1];
    transition[0][1][2] = transition[0][0][2];
    transition[0][1][3] = transition[0][0][3];
    transition[0][2][4] = (1 - model->P_discontinue) * (1 - model->P_AE_oxycodone);
    transition[0][2][5] = (1 - model->P_discontinue) * model->P_AE_oxycodone;
    transition[0][2][9] = model->P_discontinue;
    transition[0][3][4] = transition[0][2][4];
    transition[0][3][5] = transition[0][2][5];
    transition[0][3][9] = transition[0][2][9];
    transition[0][4][4] = (1 - model->P_withdrawal_AE_oxycodone - model->P_withdrawal_OR_oxycodone) * (1 - model->P_AE_oxycodone);
    transition[0][4][5] = (1 - model->P_withdrawal_AE_oxycodone - model->P_withdrawal_OR_oxycodone) * model->P_AE_oxycodone;
    transition[0][4][6] = model->P_withdrawal_AE_oxycodone;
    transition[0][4][7] = model->P_withdrawal_OR_oxycodone;
    transition[0][5][4] = transition[0][4][4];
    transition[0][5][5] = transition[0][4][5];
    transition[0][5][6] = transition[0][4][6];
    transition[0][5][7] = transition[0][4][7];
    transition[0][6][9] = 1;
    transition[0][7][9] = 1;
    transition[0][8][8] = 1;
    transition[0][9][9] = 1;

    double P_AE_novel_therapy = model->P_AE_oxycodone * 0.7;
    double P_withdrawal_AE_novel_therapy = model->P_withdrawal_AE_oxycodone * 0.7;
    double P_withdrawal_OR_novel_therapy = model->P_withdrawal_OR_oxycodone * 0.7;

    transition[1][0][0] = (1 - P_withdrawal_AE_novel_therapy - P_withdrawal_OR_novel_therapy) * (1 - P_AE_novel_therapy);
    transition[1][0][1] = (1 - P_withdrawal_AE_novel_therapy - P_withdrawal_OR_novel_therapy) * P_AE_novel_therapy;
    transition[1][0][2] = P_withdrawal_AE_novel_therapy;
    transition[1][0][3] = P_withdrawal_OR_novel_therapy;
    transition[1][1][0] = transition[1][0][0];
    transition[1][1][1] = transition[1][0][1];
    transition[1][1][2] = transition[1][0][2];
    transition[1][1][3] = transition[1][0][3];
    transition[1][2][4] = (1 - model->P_discontinue) * (1 - model->P_AE_oxycodone);
    transition[1][2][5] = (1 - model->P_discontinue) * model->P_AE_oxycodone;
    transition[1][2][9] = model->P_discontinue;
    transition[1][3][4] = transition[1][2][4];
    transition[1][3][5] = transition[1][2][5];
    transition[1][3][9] = transition[1][2][9];
    transition[1][4][4] = (1 - model->P_withdrawal_AE_oxycodone - model->P_withdrawal_OR_oxycodone) * (1 - model->P_AE_oxycodone);
    transition[1][4][5] = (1 - model->P_withdrawal_AE_oxycodone - model->P_withdrawal_OR_oxycodone) * model->P_AE_oxycodone;
    transition[1][4][6] = model->P_withdrawal_AE_oxycodone;
    transition[1][4][7] = model->P_withdrawal_OR_oxycodone;
    transition[1][5][4] = transition[1][4][4];
    transition[1][5][5] = transition[1][4][5];
    transition[1][5][6] = transition[1][4][6];
    transition[1][5][7] = transition[1][4][7];
    transition[1][6][9] = 1;
    transition[1][7][9] = 1;
    transition[1][8][8] = 1;
    transition[1][9][9] = 1;

    double C_novel_therapy = model->C_oxycodone * 0.7;
    vector < vector <double> > cost(2, vector<double>(10));

    cost[0][0] = 2.63 + model->C_morphine;
    cost[1][0] = 55.21 + C_novel_therapy;
    cost[0][1] = cost[0][0] + model->C_AE;
    cost[1][1] = cost[1][0] + model->C_AE;
    cost[0][2] = cost[1][2] = model->C_withdrawal;
    cost[0][3] = cost[1][3] = model->C_withdrawal;
    cost[0][4] = cost[1][4] = 9.20 + model->C_oxycodone;
    cost[0][5] = cost[1][5] = 9.20 + model->C_oxycodone + model->C_AE;
    cost[0][6] = cost[1][6] = model->C_withdrawal;
    cost[0][7] = cost[1][7] = model->C_withdrawal;
    cost[0][8] = cost[1][8] = 4.893;
    cost[0][9] = cost[1][9] = model->C_discontinue;

    double U_multiplier = 0.9;
    double U_discontinue = model->U_withdrawal_OR * 0.8;
    vector <double> util(10);

    util[0] = model->U_no_AE;
    util[1] = model->U_AE;
    util[2] = model->U_withdrawal_AE;
    util[3] = model->U_withdrawal_OR;
    util[4] = model->U_no_AE         * U_multiplier;
    util[5] = model->U_AE            * U_multiplier;
    util[6] = model->U_withdrawal_AE * U_multiplier;
    util[7] = model->U_withdrawal_OR * U_multiplier;
    util[8] = (model->U_withdrawal_AE + model->U_withdrawal_OR) / 2;
    util[9] = U_discontinue;

    int WTP = 20000;
    double discount = 12.5174;

    for (int i = 0; i < 2; i++) {
        double sum_of_cost = 0, sum_of_util = 0;
        vector <double> init(10);
        init[0] = 1;

        for (int t = 0; t < 51; t++) {
            vector<double> now = init * transition[i];
            vector<double> half_cycle_correction(10);
            for (int j = 0; j < 10; j++) {
                half_cycle_correction[j] = (init[j] + now[j]) / 2;
            }

            init = now;
            sum_of_cost += cost[i] * half_cycle_correction;
            sum_of_util += util * half_cycle_correction;
        }

        sum_of_cost += cost[i] * init;
        sum_of_util += util * init;

        info->val[i] = discount * (sum_of_util * WTP - sum_of_cost);
    }
}

int main() {
    MlmcInfo *info = mlmc_init(1, 2, 30, 1.0, 0.25);
    // smc_evpi_calc(info->layer[0].evppi_info, 1000000); // evpi = 1085
    mlmc_test(info, 10, 2000);

    vector <double> eps = {5, 2, 1, 0.5, 0.2, 0.1};
    mlmc_test_eval_eps(info, eps);

    return 0;
}
