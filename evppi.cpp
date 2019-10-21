
#include <iostream>
#include <iomanip>
#include <math.h>

#include "evppi.hpp"

using namespace std;

Result *result_init();

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

void evppi_calc(EvppiInfo *info, Result *result) {
    pre_sampling(info->pre);

    double max_sum = 0.0, sum_max1 = 0.0, sum_max2 = 0.0;
    for (int m = 0; m < info->m; m++) {
        post_sampling(info->post);

        double val1 = f1(info);
        double val2 = f2(info);
        max_sum += max(val1, val2);
        sum_max1 += val1;
        sum_max2 += val2;
    }

    double M = (double)(info->m);
    max_sum /= M;
    double sum_max = max(sum_max1, sum_max2) / M;

    double p = max_sum - sum_max;
    result->dp += p;
    result->dp2 += p * p;

    if (info->level) {
        double max_sum = 0.0, sum_max1 = 0.0, sum_max2 = 0.0;
        double first_max_sum = 0.0, first_sum_max1 = 0.0, first_sum_max2 = 0.0;
        double second_max_sum = 0.0, second_sum_max1 = 0.0, second_sum_max2 = 0.0;

        for (int m = 0; m < info->m / 2; m++) {
            post_sampling(info->post);

            double val1 = f1(info);
            double val2 = f2(info);

            first_max_sum += max(val1, val2);
            max_sum += max(val1, val2);

            first_sum_max1 += val1;
            first_sum_max2 += val2;

            sum_max1 += val1;
            sum_max2 += val2;
        }

        for (int m = info->m / 2; m < info->m; m++) {
            post_sampling(info->post);

            double val1 = f1(info);
            double val2 = f2(info);

            second_max_sum += max(val1, val2);
            max_sum += max(val1, val2);

            second_sum_max1 += val1;
            second_sum_max2 += val2;

            sum_max1 += val1;
            sum_max2 += val2;
        }

        double M = (double)(info->m);

        max_sum /= M;
        first_max_sum /= (M / 2);
        second_max_sum /= (M / 2);

        double sum_max = max(sum_max1, sum_max2) / M;
        double first_sum_max = max(first_sum_max1, first_sum_max2) / (M / 2);
        double second_sum_max = max(second_sum_max1, second_sum_max2) / (M / 2);

        double z = max_sum - sum_max;
        z -= (first_max_sum - first_sum_max) / 2;
        z -= (second_max_sum - second_sum_max) / 2;

        result->pf += z;
        result->pf2 += z * z;
    }
}

void mlmc_calc(MlmcInfo *info, int level, vector <int> &n_samples) {
    for (int l = 0; l <= level; l++) {
        for (int i = info->layer[l].n; i < n_samples[l]; i++) {
            evppi_calc(info->layer[l].evppi_info, info->layer[l].result);
        }

        Result *result = info->layer[l].result;
        double n = (double)n_samples[l];
        info->layer[l].aveP = result->dp / n;
        info->layer[l].aveZ = result->pf / n;
        info->layer[l].varP = result->dp2 / n - info->layer[l].aveP * info->layer[l].aveP;
        info->layer[l].varZ = result->pf2 / n - info->layer[l].aveZ * info->layer[l].aveZ;
        info->layer[l].n = n_samples[l];
    }
}

void mlmc_test(MlmcInfo *info, int test_level, int n_sample) {
    cout.precision(2);
    cout << " l  aveZ      aveP      varZ      varP" << endl;
    cout << "-------------------------------------------" << endl;

    vector <int> n_samples(test_level + 1, n_sample);
    vector <double> aveZ(test_level + 1), varZ(test_level + 1);
    for (int l = 0; l <= test_level; l++) {
        mlmc_calc(info, l, n_samples);

        aveZ[l] = info->layer[l].aveZ;
        varZ[l] = info->layer[l].varZ;

        cout << right << setw(2) << l << "  ";

        cout << scientific << aveZ[l] << "  " << info->layer[l].aveP << "  ";
        cout << varZ[l] << "  " << info->layer[l].varP << '\n';
    }

    info->alpha = log2_regression(aveZ);
    info->beta = log2_regression(varZ);

    cout << fixed << '\n';
    cout << "alpha = " << info->alpha << '\n';
    cout << "beta  = " << info->beta << "\n\n";
}

void mlmc_eval_eps(MlmcInfo *info, int level, double eps) {
    bool converged = false;
    vector <int> n_samples(info->max_level + 1);
    for (int l = 0; l <= level; l++) n_samples[l] = 1000;

    while (!converged) {
        double sum = 0.0;
        for (int l = 0; l <= level; l++) {
            mlmc_calc(info, l, n_samples);

            sum += sqrt(info->layer[l].varZ * info->layer[l].cost);
        }

        double diff = 0.0;
        for (int l = 0; l <= level; l++) {
            int add = ceil(max(0.0,
                    sqrt(info->layer[l].varZ / info->layer[l].cost) * sum / ((1.0 - info->theta) * eps * eps) - info->layer[l].n));

            n_samples[l] += add;
            diff += max(0.0, add - 0.01 * info->layer[l].n);
        }

        if (diff == 0) {
            converged = true;
            double rem = info->layer[level].aveZ / (pow(2.0, info->alpha) - 1.0);

            if (rem > sqrt(info->theta) * eps) {
                if (level == info->max_level - 1) {
                    cout << " Level over !!!" << endl;
                    exit(1);
                } else {
                    converged = false;
                    level++;
                    info->layer[level].varZ = info->layer[level - 1].varZ / pow(2.0, info->beta);

                    sum += sqrt(info->layer[level].varZ * info->layer[level].cost);
                    // sum += info->layer[level].varZ * info->layer[level].cost;
                    for (int l = 0; l <= level; l++) {
                        n_samples[l] += ceil(max(0.0,
                                sqrt(info->layer[l].varZ / info->layer[l].cost) * sum / ((1.0 - info->theta) * eps * eps) - info->layer[l].n));
                    }
                }
            }
        }
    }
}

void mlmc_test_eval_eps(MlmcInfo *info, vector <double> &eps) {
    cout << "eps       mlmc      std       save    N...\n";
    cout << "----------------------------------------------------------------\n";

    for (int i = 0; i < (int)eps.size(); i++) {
        for (int l = 0; l <= info->max_level; l++) {
            info->layer[l].n = 0;
            info->layer[l].aveZ = 0.0;
            info->layer[l].aveP = 0.0;
            info->layer[l].varZ = 0.0;
            info->layer[l].varP = 0.0;
            info->layer[l].result = result_init();
        }

        int level = 2;
        mlmc_eval_eps(info, level, eps[i]);

        for (; level < info->max_level; level++) {
            if (info->layer[level + 1].n == 0) break;
        }

        double mlmc_cost = 0.0;
        for (int l = 0; l <= level; l++) mlmc_cost += info->layer[l].n * info->layer[l].cost;
        double std_cost = info->layer[level].varP * info->layer[level].cost / ((1.0 - info->theta) * eps[i] * eps[i]);

        cout << scientific << eps[i] << "  ";
        cout << scientific << mlmc_cost << "  " << std_cost << "  ";
        cout << right << setw(4) << fixed << std_cost / mlmc_cost << "  ";
        for (int l = 0; l <= level; l++) {
            cout << info->layer[l].n << ' ';
        }
        cout << endl;
    }
}

EvppiInfo *evppi_init(int level, int m) {
    EvppiInfo *info = new EvppiInfo;
    info->level = level;
    info->m = m;
    info->pre = new PreInfo;
    pre_init(info->pre);
    info->post = new PostInfo;
    post_init(info->post);
    return info;
}

Result *result_init() {
    Result *result = new Result;
    result->dp = result->dp2 = 0;
    result->pf = result->pf2 = 0;
    return result;
}

MlmcInfo *mlmc_init(int m0, int s, int max_level, double gamma, double theta) {
    MlmcInfo *info = new MlmcInfo;
    info->max_level = max_level;
    info->alpha = info->beta = 0.0;
    info->gamma = gamma;
    info->theta = theta;
    info->layer.resize(max_level + 1);
    for (int l = 0; l <= max_level; l++) {
        info->layer[l].n = 0;
        info->layer[l].cost = pow(2.0, l * gamma);
        info->layer[l].aveZ = 0.0;
        info->layer[l].aveP = 0.0;
        info->layer[l].varZ = 0.0;
        info->layer[l].varP = 0.0;
        info->layer[l].evppi_info = evppi_init(l, m0);
        info->layer[l].result = result_init();
        m0 *= s;
    }
    return info;
}