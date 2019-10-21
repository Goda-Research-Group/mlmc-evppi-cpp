
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

normal_distribution<double> x5_dist(0.7, 0.1);
normal_distribution<double> x14_dist(0.8, 0.2);

normal_distribution<double> y1_dist(1000, 1);
normal_distribution<double> y2_dist(0.1, 0.02);
normal_distribution<double> y3_dist(5.2, 1.0);
normal_distribution<double> y4_dist(400, 200);
normal_distribution<double> y6_dist(0.3, 0.1);
normal_distribution<double> y7_dist(3, 0.5);
normal_distribution<double> y8_dist(0.25, 0.1);
normal_distribution<double> y9_dist(-0.1, 0.02);
normal_distribution<double> y10_dist(0.5, 0.2);
normal_distribution<double> y11_dist(1500, 1);
normal_distribution<double> y12_dist(0.08, 0.02);
normal_distribution<double> y13_dist(6.1, 1.0);
normal_distribution<double> y15_dist(0.3, 0.05);
normal_distribution<double> y16_dist(3.0, 1.0);
normal_distribution<double> y17_dist(0.2, 0.05);
normal_distribution<double> y18_dist(-0.1, 0.02);
normal_distribution<double> y19_dist(0.5, 0.2);

void pre_init(PreInfo *info) {
    info->x.resize(20);
}

void post_init(PostInfo *info) {
    info->y.resize(20);
}

// TODO
// X5, X7, X14, X16 are pairwise correlated with a correlation coefficient ρ = 0.6

void pre_sampling(PreInfo *info) {
    info->x[5] = x5_dist(generator);
    info->x[14] = x14_dist(generator);
}

void post_sampling(PostInfo *info) {
    info->y[1] = y1_dist(generator);
    info->y[2] = y2_dist(generator);
    info->y[3] = y3_dist(generator);
    info->y[4] = y4_dist(generator);
    info->y[6] = y6_dist(generator);
    info->y[7] = y7_dist(generator);
    info->y[8] = y8_dist(generator);
    info->y[9] = y9_dist(generator);
    info->y[10] = y10_dist(generator);
    info->y[11] = y11_dist(generator);
    info->y[12] = y12_dist(generator);
    info->y[13] = y13_dist(generator);
    info->y[15] = y15_dist(generator);
    info->y[16] = y16_dist(generator);
    info->y[17] = y17_dist(generator);
    info->y[18] = y18_dist(generator);
    info->y[19] = y19_dist(generator);
}

double f1(EvppiInfo *info) {
    double tmp1 = info->pre->x[5] * info->post->y[6] * info->post->y[7];
    double tmp2 = info->post->y[8] * info->post->y[9] * info->post->y[10];
    double tmp3 = info->post->y[1] + info->post->y[2] * info->post->y[3] * info->post->y[4];
    return lambda * (tmp1 + tmp2) - tmp3;
}

double f2(EvppiInfo *info) {
    double tmp1 = info->pre->x[14] * info->post->y[15] * info->post->y[16];
    double tmp2 = info->post->y[17] * info->post->y[18] * info->post->y[19];
    double tmp3 = info->post->y[11] + info->post->y[12] * info->post->y[13] * info->post->y[4];
    return lambda * (tmp1 + tmp2) - tmp3;
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