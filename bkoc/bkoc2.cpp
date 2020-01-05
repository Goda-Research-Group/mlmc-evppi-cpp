
#include <random>
#include <iostream>

#include "../evppi.hpp"
#include "../matrix.hpp"

using namespace std;

random_device rd;
mt19937 generator(rd());

int m0 = 1;
int s = 2;
int max_level = 20;
int test_level = 10;
int n_sample = 2000;
int lambda = 10000;

vector <double> u(4), post_u(2);
Matrix sigma(4, 4), post_sigma(2, 2);
Matrix sigma2_inv_sigma4(2, 2);

normal_distribution<double> dist_normal(0.0, 1.0);
normal_distribution<double> dist1(1000, 1);
normal_distribution<double> dist2(0.1, 0.02);
normal_distribution<double> dist3(5.2, 1.0);
normal_distribution<double> dist4(400, 200);
normal_distribution<double> dist6(0.3, 0.1);
normal_distribution<double> dist8(0.25, 0.1);
normal_distribution<double> dist9(-0.1, 0.02);
normal_distribution<double> dist10(0.5, 0.2);
normal_distribution<double> dist11(1500, 1);
normal_distribution<double> dist12(0.08, 0.02);
normal_distribution<double> dist13(6.1, 1.0);
normal_distribution<double> dist15(0.3, 0.05);
normal_distribution<double> dist17(0.2, 0.05);
normal_distribution<double> dist18(-0.1, 0.02);
normal_distribution<double> dist19(0.5, 0.2);

void sampling_init(EvppiInfo *info) {
    info->model_num = 2;
    info->sample.resize(20);
    info->val.resize(info->model_num);
}

void pre_sampling(EvppiInfo *info) {
    vector <double> rand(4), ret(4);
    rand[0] = dist_normal(generator);
    rand[1] = dist_normal(generator);
    rand[2] = dist_normal(generator);
    rand[3] = dist_normal(generator);
    rand_multinormal(u, sigma, rand, ret);

    info->sample[5] = ret[0];
    info->sample[6] = dist6(generator);
    info->sample[14] = ret[2];
    info->sample[15] = dist15(generator);

    Matrix diff(2, 1);
    diff[0][0] = ret[0] - u[0];
    diff[1][0] = ret[2] - u[2];

    Matrix tmp = sigma2_inv_sigma4 * diff;
    post_u[0] = u[1] + tmp[0][0];
    post_u[1] = u[3] + tmp[1][0];
}

void post_sampling(EvppiInfo *info) {
    vector <double> rand(2), ret(2);
    rand[0] = dist_normal(generator);
    rand[1] = dist_normal(generator);
    rand_multinormal(post_u, post_sigma, rand, ret);

    info->sample[1] = dist1(generator);
    info->sample[2] = dist2(generator);
    info->sample[3] = dist3(generator);
    info->sample[4] = dist4(generator);
    info->sample[7] = ret[0];
    info->sample[8] = dist8(generator);
    info->sample[9] = dist9(generator);
    info->sample[10] = dist10(generator);
    info->sample[11] = dist11(generator);
    info->sample[12] = dist12(generator);
    info->sample[13] = dist13(generator);
    info->sample[16] = ret[1];
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

void matrix_init() {
    double rho = 0.6;
    u = {0.7, 3.0, 0.8, 3.0};
    double r5 = 0.1, r7 = 0.5, r14 = 0.1, r16 = 1.0;

    sigma[0] = { r5 * r5,        r5 * r7 * rho,  r5 * r14 * rho,  r5 * r16 * rho  };
    sigma[1] = { r7 * r5 * rho,  r7 * r7,        r7 * r14 * rho,  r7 * r16 * rho  };
    sigma[2] = { r14 * r5 * rho, r14 * r7 * rho, r14 * r14,       r14 * r16 * rho };
    sigma[3] = { r16 * r5 * rho, r16 * r7 * rho, r16 * r14 * rho, r16 * r16       };
    sigma = Cholesky(sigma);

    Matrix sigma1(2, 2);
    sigma1[0] = { r7 * r7,        r7 * r16 * rho };
    sigma1[1] = { r16 * r7 * rho, r16 * r16      };

    Matrix sigma2(2, 2);
    sigma2[0] = { r7  * r5  * rho, r7  * r14 * rho };
    sigma2[1] = { r16 * r5  * rho, r16 * r14 * rho };

    Matrix sigma3(2, 2);
    sigma3[0] = { r5 * r7 * rho,  r5 * r16 * rho  };
    sigma3[1] = { r14 * r7 * rho, r14 * r16 * rho };

    Matrix sigma4(2, 2);
    sigma4[0] = { r5 * r5,        r5 * r14 * rho };
    sigma4[1] = { r14 * r5 * rho, r14 * r14      };
    Matrix inv_sigma4 = Inverse(sigma4);

    sigma2_inv_sigma4 = sigma2 * inv_sigma4;
    Matrix tmp = sigma2_inv_sigma4 * sigma3;
    post_sigma = sigma1 - tmp;
    post_sigma = Cholesky(post_sigma);
}

int main() {
    matrix_init();

    MlmcInfo *info = mlmc_init(m0, s, max_level, 1.0, 0.25);
    mlmc_test(info, test_level, n_sample);

    // smc_evpi_calc(info->layer[0].evppi_info, 10000000);

    vector <double> eps = {2.0, 1.0, 0.5, 0.2, 0.1};
    mlmc_test_eval_eps(info, eps);

    return 0;
}