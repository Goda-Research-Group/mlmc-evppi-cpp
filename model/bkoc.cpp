
#include <random>

#include "../evppi.hpp"

using namespace std;

random_device rd;
mt19937 generator(rd());

int m0 = 1;
int s = 2;
int max_level = 20;
int test_level = 10;
int n_sample = 2000;
int lambda = 10000;

normal_distribution<double> dist5(0.7, 0.1);
normal_distribution<double> dist14(0.8, 0.2);

normal_distribution<double> dist1(1000, 1);
normal_distribution<double> dist2(0.1, 0.02);
normal_distribution<double> dist3(5.2, 1.0);
normal_distribution<double> dist4(400, 200);
normal_distribution<double> dist6(0.3, 0.1);
normal_distribution<double> dist7(3, 0.5);
normal_distribution<double> dist8(0.25, 0.1);
normal_distribution<double> dist9(-0.1, 0.02);
normal_distribution<double> dist10(0.5, 0.2);
normal_distribution<double> dist11(1500, 1);
normal_distribution<double> dist12(0.08, 0.02);
normal_distribution<double> dist13(6.1, 1.0);
normal_distribution<double> dist15(0.3, 0.05);
normal_distribution<double> dist16(3.0, 1.0);
normal_distribution<double> dist17(0.2, 0.05);
normal_distribution<double> dist18(-0.1, 0.02);
normal_distribution<double> dist19(0.5, 0.2);

void sampling_init(EvppiInfo *info) {
    info->model_num = 2;
    info->sample.resize(20);
    info->val.resize(info->model_num);
}

// TODO
// X5, X7, X14, X16 are pairwise correlated with a correlation coefficient ρ = 0.6

void pre_sampling(EvppiInfo *info) {
    info->sample[5] = dist5(generator);
    info->sample[14] = dist14(generator);
}

void post_sampling(EvppiInfo *info) {
    info->sample[1] = dist1(generator);
    info->sample[2] = dist2(generator);
    info->sample[3] = dist3(generator);
    info->sample[4] = dist4(generator);
    info->sample[6] = dist6(generator);
    info->sample[7] = dist7(generator);
    info->sample[8] = dist8(generator);
    info->sample[9] = dist9(generator);
    info->sample[10] = dist10(generator);
    info->sample[11] = dist11(generator);
    info->sample[12] = dist12(generator);
    info->sample[13] = dist13(generator);
    info->sample[15] = dist15(generator);
    info->sample[16] = dist16(generator);
    info->sample[17] = dist17(generator);
    info->sample[18] = dist18(generator);
    info->sample[19] = dist19(generator);
}

void f(EvppiInfo *info) {
    double tmp1, tmp2, tmp3;
    
    tmp1 = info->sample[5] * info->sample[6] * info->sample[7];
    tmp2 = info->sample[8] * info->sample[9] * info->sample[10];
    tmp3 = info->sample[1] + info->sample[2] * info->sample[3] * info->sample[4];
    info->val[0] = lambda * (tmp1 + tmp2) - tmp3;

    tmp1 = info->sample[14] * info->sample[15] * info->sample[16];
    tmp2 = info->sample[17] * info->sample[18] * info->sample[19];
    tmp3 = info->sample[11] + info->sample[12] * info->sample[13] * info->sample[4];
    info->val[1] = lambda * (tmp1 + tmp2) - tmp3;
}

int main() {
    MlmcInfo *info = mlmc_init(m0, s, max_level, 1.0, 0.25);
    mlmc_test(info, test_level, n_sample);

    vector <double> eps;
    eps.push_back(2.0);
    eps.push_back(1.0);
    eps.push_back(0.5);
    eps.push_back(0.2);
    eps.push_back(0.1);

    mlmc_test_eval_eps(info, eps);

    return 0;
}