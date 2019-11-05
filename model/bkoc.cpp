
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

double rho = 0.6;
vector <double> u(4), post_u(2);
Matrix sigma(4, 4), sigma_cholesky(4, 4);
Matrix sigma1(2, 2), sigma2(2, 2), sigma3(2, 2), sigma4(2, 2), inv_sigma4(2, 2);
Matrix post_sigma(2, 2), post_sigma_cholesky(2, 2);

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
    vector <double> ran(4);
    ran[0] = dist_normal(generator);
    ran[1] = dist_normal(generator);
    ran[2] = dist_normal(generator);
    ran[3] = dist_normal(generator);
    vector <double> r = rand_multinormal(u, sigma_cholesky, ran);

    info->sample[5] = r[0];
    info->sample[14] = r[2];

    Matrix diff(2, 1);
    diff[0][0] = r[0] - u[0];
    diff[1][0] = r[2] - u[2];

    Matrix tmp1 = sigma2 * inv_sigma4;
    Matrix tmp2 = tmp1 * diff;
    post_u[0] = u[1] + tmp2[0][0];
    post_u[1] = u[3] + tmp2[1][0];
}

void post_sampling(EvppiInfo *info) {
    vector <double> ran(2);
    ran[0] = dist_normal(generator);
    ran[1] = dist_normal(generator);
    vector <double> r = rand_multinormal(post_u, post_sigma_cholesky, ran);

    info->sample[1] = dist1(generator);
    info->sample[2] = dist2(generator);
    info->sample[3] = dist3(generator);
    info->sample[4] = dist4(generator);
    info->sample[6] = dist6(generator);
    info->sample[7] = r[0];
    info->sample[8] = dist8(generator);
    info->sample[9] = dist9(generator);
    info->sample[10] = dist10(generator);
    info->sample[11] = dist11(generator);
    info->sample[12] = dist12(generator);
    info->sample[13] = dist13(generator);
    info->sample[15] = dist15(generator);
    info->sample[16] = r[1];
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
    u = {0.7, 3.0, 0.8, 3.0};
    double r5 = 0.1, r7 = 0.5, r14 = 0.1, r16 = 1.0;

    sigma[0][0] = r5  * r5;
    sigma[0][1] = r5  * r7  * rho;
    sigma[0][2] = r5  * r14 * rho;
    sigma[0][3] = r5  * r16 * rho;
    sigma[1][0] = r7  * r5  * rho;
    sigma[1][1] = r7  * r7;
    sigma[1][2] = r7  * r14 * rho;
    sigma[1][3] = r7  * r16 * rho;
    sigma[2][0] = r14 * r5  * rho;
    sigma[2][1] = r14 * r7  * rho;
    sigma[2][2] = r14 * r14;
    sigma[2][3] = r14 * r16 * rho;
    sigma[3][0] = r16 * r5  * rho;
    sigma[3][1] = r16 * r7  * rho;
    sigma[3][2] = r16 * r14 * rho;
    sigma[3][3] = r16 * r16;
    sigma_cholesky = Cholesky(sigma);

    sigma1[0][0] = r7  * r7;
    sigma1[0][1] = r7  * r16 * rho;
    sigma1[1][0] = r16 * r7  * rho;
    sigma1[1][1] = r16 * r16;

    sigma2[0][0] = r7  * r5  * rho;
    sigma2[0][1] = r7  * r14 * rho;
    sigma2[1][0] = r16 * r5  * rho;
    sigma2[1][1] = r16 * r14 * rho;

    sigma3[0][0] = r5  * r7  * rho;
    sigma3[0][1] = r5  * r16 * rho;
    sigma3[1][0] = r14 * r7  * rho;
    sigma3[1][1] = r14 * r16 * rho;

    sigma4[0][0] = r5  * r5;
    sigma4[0][1] = r5  * r14 * rho;
    sigma4[1][0] = r14 * r5  * rho;
    sigma4[1][1] = r14 * r14;
    inv_sigma4 = Inverse(sigma4);

    Matrix tmp1 = sigma2 * inv_sigma4;
    Matrix tmp2 = tmp1 * sigma3;
    post_sigma = sigma1 - tmp2;
    post_sigma_cholesky = Cholesky(post_sigma);

    MlmcInfo *info = mlmc_init(m0, s, max_level, 1.0, 0.25);
    mlmc_test(info, test_level, n_sample);

    // smc_evpi_calc(info->layer[0].evppi_info, 10000000);

    vector <double> eps = {2.0, 1.0, 0.5, 0.2, 0.1};
    mlmc_test_eval_eps(info, eps);

    return 0;
}