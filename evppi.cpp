
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <algorithm>

#include "evppi.hpp"
#include "util.hpp"

using namespace std;

Result *result_init();

void evppi_calc(EvppiInfo *info, Result *result) {
    pre_sampling(info);

    vector <double> sum(info->model_num);
    double sum_of_max = 0.0;
    for (int m = 0; m < info->m; m++) {
        post_sampling(info);

        f(info);
        double mx = -1e30;
        for (int i = 0; i < info->model_num; i++) {
            mx = max(mx, info->val[i]);
            sum[i] += info->val[i];
        }
        sum_of_max += mx;
    }

    double M = (double)(info->m);
    sum_of_max /= M;
    double max_of_sum = *max_element(sum.begin(), sum.end()) / M;

    double p = sum_of_max - max_of_sum;
    result->p1 += p;
    result->p2 += p * p;

    if (info->level) {
        vector <double> sum(info->model_num);
        vector <double> sum_a(info->model_num), sum_b(info->model_num);
        for (int m = 0; m < info->m / 2; m++) {
            post_sampling(info);

            f(info);
            for (int i = 0; i < info->model_num; i++) {
                sum[i] += info->val[i];
                sum_a[i] += info->val[i];
            }
        }

        for (int m = info->m / 2; m < info->m; m++) {
            post_sampling(info);

            f(info);
            for (int i = 0; i < info->model_num; i++) {
                sum[i] += info->val[i];
                sum_b[i] += info->val[i];
            }
        }

        double max_of_sum = *max_element(sum.begin(), sum.end()) / (double)(info->m);
        double max_of_sum_a = *max_element(sum_a.begin(), sum_a.end()) / (double)(info->m / 2);
        double max_of_sum_b = *max_element(sum_b.begin(), sum_b.end()) / (double)(info->m - info->m / 2);

        double z = (max_of_sum_a + max_of_sum_b) / 2.0 - max_of_sum;
        result->z1 += z;
        result->z2 += z * z;
        result->z3 += z * z * z;
        result->z4 += z * z * z * z;
    }
}

void mlmc_calc(MlmcInfo *info, int level, vector <int> &n_samples) {
    for (int l = 0; l <= level; l++) {
        for (int i = info->layer[l].n; i < n_samples[l]; i++) {
            evppi_calc(info->layer[l].evppi_info, info->layer[l].result);
        }

        Result *result = info->layer[l].result;
        double n = (double)n_samples[l];
        result->p1 /= n;
        result->p2 /= n;
        result->z1 /= n;
        result->z2 /= n;
        result->z3 /= n;
        result->z4 /= n;

        info->layer[l].aveP = result->p1;
        info->layer[l].aveZ = result->z1;
        info->layer[l].varP = result->p2 - result->p1 * result->p1;
        info->layer[l].varZ = result->z2 - result->z1 * result->z1;
        if (l) {
            info->layer[l].kurt =
                    ((result->z4 - 4 * result->z3 * result->z1 + 6 * result->z2 * result->z1 * result->z1 -
                      3 * result->z1 * result->z1 * result->z1 * result->z1) /
                     ((result->z2 - result->z1 * result->z1) * (result->z2 - result->z1 * result->z1)));
        }
        info->layer[l].n = n_samples[l];
    }
}

void mlmc_test(MlmcInfo *info, int test_level, int n_sample, const char *file_name) {
    cout.precision(2);
    cout << " l  aveZ      aveP      varZ      varP      kurt\n";
    cout << "----------------------------------------------------\n";

    ofstream ofs(file_name, ios::out);
    ofs << " l   aveZ      aveP      varZ      varP      kurt\n";
    ofs << "-----------------------------------------------------\n";

    vector <int> n_samples(test_level + 1, n_sample);
    vector <double> aveZ(test_level + 1), varZ(test_level + 1);
    for (int l = 0; l <= test_level; l++) {
        mlmc_calc(info, l, n_samples);

        aveZ[l] = info->layer[l].aveZ;
        varZ[l] = info->layer[l].varZ;

        cout << right << setw(2) << l << "  ";
        cout << scientific << aveZ[l] << "  " << info->layer[l].aveP << "  ";
        cout << varZ[l] << "  " << info->layer[l].varP << "  " << info->layer[l].kurt << '\n';

        ofs << right << setw(2) << l << "  ";
        ofs << scientific << aveZ[l] << "  " << info->layer[l].aveP << "  ";
        ofs << varZ[l] << "  " << info->layer[l].varP << "  " << info->layer[l].kurt << '\n';
    }

    info->alpha = log2_regression(aveZ);
    info->beta = log2_regression(varZ);

    cout << fixed << '\n';
    cout << "alpha = " << info->alpha << '\n';
    cout << "beta  = " << info->beta << '\n' << endl;

    ofs << fixed << '\n';
    ofs << "alpha = " << info->alpha << '\n';
    ofs << "beta  = " << info->beta << '\n' << endl;
    ofs.close();
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

void mlmc_test_eval_eps(MlmcInfo *info, vector <double> &eps, const char *file_name) {
    cout << "eps       mlmc      std       save    N...\n";
    cout << "----------------------------------------------------------------\n";

    ofstream ofs(file_name, ios::app);
    ofs << "eps       mlmc      std       save    N...\n";
    ofs << "----------------------------------------------------------------\n";

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

        ofs << scientific << eps[i] << "  ";
        ofs << scientific << mlmc_cost << "  " << std_cost << "  ";
        ofs << right << setw(4) << fixed << std_cost / mlmc_cost << "  ";

        for (int l = 0; l <= level; l++) {
            cout << info->layer[l].n << ' ';
            ofs << info->layer[l].n << ' ';
        }

        cout << endl;
        ofs << endl;
    }

    ofs.close();
}

EvppiInfo *evppi_init(int level, int m) {
    EvppiInfo *info = new EvppiInfo;
    info->level = level;
    info->m = m;
    sampling_init(info);
    return info;
}

Result *result_init() {
    Result *result = new Result;
    result->p1 = result->p2 = 0;
    result->z1 = result->z2 = result->z3 = result->z4 = 0;
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
        info->layer[l].kurt = 0.0;
        info->layer[l].evppi_info = evppi_init(l, m0);
        info->layer[l].result = result_init();
        m0 *= s;
    }
    return info;
}