
#include <random>

#include "../evppi.hpp"

using namespace std;

random_device rd;
mt19937 generator(rd());

int m0 = 1;
int s = 2;
int max_level = 20;
int test_level = 10;
int n_sample = 20000;

normal_distribution<double> x_dist(0.0, 1.0);
normal_distribution<double> y_dist(0.0, 1.0);

void sampling_init(EvppiInfo *info) {
    info->model_num = 2;
    info->sample.resize(2);
    info->val.resize(info->model_num);
}

void pre_sampling(EvppiInfo *info) {
    info->sample[0] = x_dist(generator);
}

void post_sampling(EvppiInfo *info) {
    info->sample[1] = y_dist(generator);
}

void f(EvppiInfo *info) {
    info->val[0] = 0.0;
    info->val[1] = info->sample[0] + info->sample[1];
}

int main() {
    MlmcInfo *info = mlmc_init(m0, s, max_level, 1.0, 0.25);
    mlmc_test(info, test_level, n_sample);

    // smc_evpi_calc(info->layer[0].evppi_info, 10000000);

    vector <double> eps = {0.002, 0.001, 0.0005, 0.0002, 0.0001};
    mlmc_test_eval_eps(info, eps);

    return 0;
}