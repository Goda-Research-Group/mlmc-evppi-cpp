
#include <random>

#include "../evppi.hpp"

using namespace std;

random_device rd;
mt19937 generator(rd());

normal_distribution<double> x_dist(0.0, 1.0);
normal_distribution<double> y_dist(0.0, 1.0);

struct ModelInfo {
    double x, y;
};

void sampling_init(EvppiInfo *info) {
    info->model_num = 2;
    info->model_info = new ModelInfo;
    info->val.resize(info->model_num);
}

void pre_sampling(ModelInfo *info) {
    info->x = x_dist(generator);
}

void post_sampling(ModelInfo *info) {
    info->y = y_dist(generator);
}

void f(EvppiInfo *info) {
    ModelInfo *model = info->model_info;
    info->val[0] = 0.0;
    info->val[1] = model->x + model->y;
}

int main() {
    MlmcInfo *info = mlmc_init(1, 2, 30, 1.0, 0.25);
    // smc_evpi_calc(info->layer[0].evppi_info, 10000000);
    mlmc_test(info, 10, 200000);

    vector <double> eps = {0.002, 0.001, 0.0005, 0.0002, 0.0001};
    mlmc_test_eval_eps(info, eps);

    return 0;
}