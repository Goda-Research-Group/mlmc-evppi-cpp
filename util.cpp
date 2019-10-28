
#include <vector>
#include <math.h>

#include "util.hpp"

double expit(double x) {
    return exp(x) / (1 + exp(x));
}

double logit(double x) {
    return log(x / (1 - x));
}

double regression(vector <double> &x, vector <double> &y) {
    double n = x.size();
    double sum_x = 0.0, sum_y = 0.0, sum_x_x = 0.0, sum_x_y = 0.0;
    for (int l = 0; l < n; l++) {
        sum_x += x[l];
        sum_y += y[l];
        sum_x_x += x[l] * x[l];
        sum_x_y += x[l] * y[l];
    }
    return (n * sum_x_y - sum_x * sum_y) / (n * sum_x_x - sum_x * sum_x);
}

double log2_regression(vector <double> &y) {
    int n = y.size() - 1;
    vector<double> x(n), log2_y(n);
    for (int l = 1; l <= n; l++) {
        x[l - 1] = l;
        log2_y[l - 1] = -log2(y[l]);
    }
    return regression(x, log2_y);
}