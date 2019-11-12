
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

normal_distribution<double> dist(0.0, 1.0);

void sampling_init(EvppiInfo *info) {
    info->model_num = 2;
    info->sample.resize(2);
    info->val.resize(info->model_num);
}

void pre_sampling(EvppiInfo *info) {
    double r = dist(generator);
    info->sample[0] = r >= 0.0 ? 1 : -1;
}

void post_sampling(EvppiInfo *info) {
    double r = dist(generator);
    info->sample[1] = r >= 0.0 ? 1 : -1;
}

void f(EvppiInfo *info) {
    info->val[0] = info->sample[0] * info->sample[1];
    info->val[1] = - info->sample[0] * info->sample[1];
}

int main() {
    MlmcInfo *info = mlmc_init(m0, s, max_level, 1.0, 0.25);
    smc_evpi_calc(info->layer[0].evppi_info, 10000000);
    mlmc_test(info, test_level, n_sample);

    return 0;
}