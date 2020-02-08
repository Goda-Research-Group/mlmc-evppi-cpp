
#include <random>
#include <iostream>

#include "../evppi.hpp"
#include "../matrix.hpp"

using namespace std;

random_device rd;
mt19937 generator(rd());

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

struct ModelInfo {
    double x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19;
};

void sampling_init(EvppiInfo *info) {
    info->model_num = 2;
    info->model_info = new ModelInfo;
    info->val.resize(info->model_num);
}

void pre_sampling(ModelInfo *model) {
    vector <double> rand(4), ret(4);
    rand[0] = dist_normal(generator);
    rand[1] = dist_normal(generator);
    rand[2] = dist_normal(generator);
    rand[3] = dist_normal(generator);
    rand_multinormal(u, sigma, rand, ret);

    model->x7 = ret[1];
    model->x16 = ret[3];

    Matrix diff(2, 1);
    diff[0][0] = ret[1] - u[1];
    diff[1][0] = ret[3] - u[3];

    Matrix tmp = sigma2_inv_sigma4 * diff;
    post_u[0] = u[0] + tmp[0][0];
    post_u[1] = u[2] + tmp[1][0];
}

void post_sampling(ModelInfo *model) {
    vector <double> rand(2), ret(2);
    rand[0] = dist_normal(generator);
    rand[1] = dist_normal(generator);
    rand_multinormal(post_u, post_sigma, rand, ret);

    model->x5 = ret[0];
    model->x14 = ret[1];

    model->x1 = dist1(generator);
    model->x2 = dist2(generator);
    model->x3 = dist3(generator);
    model->x4 = dist4(generator);
    model->x6 = dist6(generator);
    model->x8 = dist8(generator);
    model->x9 = dist9(generator);
    model->x10 = dist10(generator);
    model->x11 = dist11(generator);
    model->x12 = dist12(generator);
    model->x13 = dist13(generator);
    model->x15 = dist15(generator);
    model->x17 = dist17(generator);
    model->x18 = dist18(generator);
    model->x19 = dist19(generator);
}

void f(EvppiInfo *info) {
    ModelInfo *model = info->model_info;

    double tmp1, tmp2, tmp3;
    int lambda = 10000;

    tmp1 = model->x5 * model->x6 * model->x7;
    tmp2 = model->x8 * model->x9 * model->x10;
    tmp3 = model->x1 + model->x2 * model->x3 * model->x4;
    info->val[0] = lambda * (tmp1 + tmp2) - tmp3;

    tmp1 = model->x14 * model->x15 * model->x16;
    tmp2 = model->x17 * model->x18 * model->x19;
    tmp3 = model->x11 + model->x12 * model->x13 * model->x4;
    info->val[1] = lambda * (tmp1 + tmp2) - tmp3;
}

void matrix_init() {
    u = {0.7, 3.0, 0.8, 3.0};

    double rho = 0.6;
    double r5 = 0.1, r7 = 0.5, r14 = 0.1, r16 = 1.0;

    sigma[0] = { r5 * r5,        r5 * r7 * rho,  r5 * r14 * rho,  r5 * r16 * rho  };
    sigma[1] = { r7 * r5 * rho,  r7 * r7,        r7 * r14 * rho,  r7 * r16 * rho  };
    sigma[2] = { r14 * r5 * rho, r14 * r7 * rho, r14 * r14,       r14 * r16 * rho };
    sigma[3] = { r16 * r5 * rho, r16 * r7 * rho, r16 * r14 * rho, r16 * r16       };
    sigma = Cholesky(sigma);

    Matrix sigma1(2, 2);
    sigma1[0] = { r5 * r5,        r5 * r14 * rho };
    sigma1[1] = { r14 * r5 * rho, r14 * r14      };

    Matrix sigma2(2, 2);
    sigma2[0] = { r5  * r7  * rho, r5  * r16 * rho };
    sigma2[1] = { r14 * r7  * rho, r14 * r16 * rho };

    Matrix sigma3(2, 2);
    sigma3[0] = { r7 * r5 * rho,  r7 * r14 * rho  };
    sigma3[1] = { r16 * r5 * rho, r16 * r14 * rho };

    Matrix sigma4(2, 2);
    sigma4[0] = { r7 * r7,        r7 * r16 * rho };
    sigma4[1] = { r16 * r7 * rho, r16 * r16      };
    Matrix inv_sigma4 = Inverse(sigma4);

    sigma2_inv_sigma4 = sigma2 * inv_sigma4;
    Matrix tmp = sigma2_inv_sigma4 * sigma3;
    post_sigma = sigma1 - tmp;
    post_sigma = Cholesky(post_sigma);
}

int main() {
    matrix_init();

    MlmcInfo *info = mlmc_init(1, 2, 30, 1.0, 0.25);
    // smc_evpi_calc(info->layer[0].evppi_info, 1000000);
    mlmc_test(info, 10, 2000);

    vector <double> eps = {2.0, 1.0, 0.5, 0.2, 0.1};
    mlmc_test_eval_eps(info, eps);

    return 0;
}