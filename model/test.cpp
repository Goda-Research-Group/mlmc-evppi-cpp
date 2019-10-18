
#include "../evppi.hpp"

int m0 = 1;
int s = 2;
int max_level = 20;
int test_level = 10;
double gamma = 1.0;
double theta = 0.25;
int n_sample = 200000;

double f(double x, double y) {
    return x + y;
}

int main() {
    MlmcInfo *info = mlmc_init(m0, s, max_level, gamma, theta);
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