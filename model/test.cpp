
#include <random>

#include "../evppi.hpp"

using namespace std;

random_device rd;
mt19937 generator(rd());

int m0 = 1;
int s = 2;
int max_level = 20;
int test_level = 10;
int n_sample = 200000;

normal_distribution<double> x_dist(0.0, 1.0);
normal_distribution<double> y_dist(0.0, 1.0);

void pre_init(PreInfo *info) {
    info->x.resize(1);
}

void post_init(PostInfo *info) {
    info->y.resize(1);
}

void pre_sampling(PreInfo *info) {
    info->x[0] = x_dist(generator);
}

void post_sampling(PostInfo *info) {
    info->y[0] = y_dist(generator);
}

double f1(EvppiInfo *info) {
    return 0.0;
}

double f2(EvppiInfo *info) {
    return info->pre->x[0] + info->post->y[0];
}

int main() {
    MlmcInfo *info = mlmc_init(m0, s, max_level, 1.0, 0.25);
    mlmc_test(info, test_level, n_sample);

    vector <double> eps;
    eps.push_back(0.005);
    eps.push_back(0.002);
    eps.push_back(0.001);
    eps.push_back(0.0005);
    eps.push_back(0.0002);
    eps.push_back(0.0001);

    mlmc_test_eval_eps(info, eps);

    return 0;
}