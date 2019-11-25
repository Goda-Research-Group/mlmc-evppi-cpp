
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

    double sum_of_max = 0.0;
    vector <double> sum(info->model_num);

    if (info->level == 0) {
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
        result->z1 += p;
        result->z2 += p * p;
        result->z3 += p * p * p;
        result->z4 += p * p * p * p;
    } else {
        vector <double> sum_a(info->model_num), sum_b(info->model_num);

        for (int m = 0; m < info->m / 2; m++) {
            post_sampling(info);
            f(info);

            double mx = -1e30;
            for (int i = 0; i < info->model_num; i++) {
                mx = max(mx, info->val[i]);
                sum[i] += info->val[i];
                sum_a[i] += info->val[i];
            }
            sum_of_max += mx;
        }

        for (int m = info->m / 2; m < info->m; m++) {
            post_sampling(info);
            f(info);

            double mx = -1e30;
            for (int i = 0; i < info->model_num; i++) {
                mx = max(mx, info->val[i]);
                sum[i] += info->val[i];
                sum_b[i] += info->val[i];
            }
            sum_of_max += mx;
        }

        sum_of_max /= (double)(info->m);
        double max_of_sum = *max_element(sum.begin(), sum.end()) / (double)(info->m);
        double max_of_sum_a = *max_element(sum_a.begin(), sum_a.end()) / (double)(info->m / 2);
        double max_of_sum_b = *max_element(sum_b.begin(), sum_b.end()) / (double)(info->m - info->m / 2);

        double p = sum_of_max - max_of_sum;
        result->p1 += p;
        result->p2 += p * p;

        double z = (max_of_sum_a + max_of_sum_b) / 2.0 - max_of_sum;

        // zがゼロに近いとき、数値誤差でおかしな値になっていそう。
        if (abs(z) > abs(max_of_sum) * 1e-10) {
            result->z1 += z;
            result->z2 += z * z;
            result->z3 += z * z * z;
            result->z4 += z * z * z * z;
        }
    }
}

void mlmc_calc(MlmcInfo *info, int level, vector <int> &n_samples) {
    for (int l = 0; l <= level; l++) {
        clock_t start_time = clock();
        for (int i = 0; i < n_samples[l]; i++) {
            evppi_calc(info->layer[l].evppi_info, info->layer[l].result);
        }
        clock_t end_time = clock();

        info->layer[l].n += n_samples[l];
        double n = (double)info->layer[l].n;

        Result *result = info->layer[l].result;
        result->time += end_time - start_time;
        info->layer[l].time = result->time / n;
        info->layer[l].aveP = result->p1 / n;
        info->layer[l].aveZ = result->z1 / n;
        info->layer[l].varP = result->p2 / n - info->layer[l].aveP * info->layer[l].aveP;
        info->layer[l].varZ = result->z2 / n - info->layer[l].aveZ * info->layer[l].aveZ;
        if (l) {
            info->layer[l].kurt =
                    ((result->z4 / n - 4 * result->z3 / n * info->layer[l].aveZ + 6 * result->z2 / n * info->layer[l].aveZ * info->layer[l].aveZ -
                      3 * info->layer[l].aveZ * info->layer[l].aveZ * info->layer[l].aveZ * info->layer[l].aveZ) /
                     ((result->z2 / n - info->layer[l].aveZ * info->layer[l].aveZ) * (result->z2 / n - info->layer[l].aveZ * info->layer[l].aveZ)));
            info->layer[l].check =
                    abs(info->layer[l].aveZ + info->layer[l - 1].aveP - info->layer[l].aveP) /
                    (3.0 * (sqrt(info->layer[l].varZ) + sqrt(info->layer[l - 1].varP) + sqrt(info->layer[l].varP)) / sqrt(n));

        }
    }
}

void mlmc_test(MlmcInfo *info, int test_level, int n_sample, const char *file_name) {
    cout << " l  aveZ       aveP       varZ       varP       kurt       check\n";
    cout << "--------------------------------------------------------------------\n";

    ofstream ofs(file_name, ios::out);
    ofs << " l  aveZ       aveP       varZ       varP       kurt       check\n";
    ofs << "--------------------------------------------------------------------\n";

    vector <int> n_samples(test_level + 1, n_sample);
    vector <double> aveZ(test_level + 1), varZ(test_level + 1);
    for (int l = 0; l <= test_level; l++) {
        mlmc_calc(info, l, n_samples);

        aveZ[l] = info->layer[l].aveZ;
        varZ[l] = info->layer[l].varZ;

        cout << right << setw(2) << l << "  ";
        cout << scientific << setprecision(3) << aveZ[l] << "  " << info->layer[l].aveP << "  ";
        cout << varZ[l] << "  " << info->layer[l].varP << "  " << info->layer[l].kurt << "  " << info->layer[l].check << '\n';

        ofs << right << setw(2) << l << "  ";
        ofs << scientific << setprecision(3) << aveZ[l] << "  " << info->layer[l].aveP << "  ";
        ofs << varZ[l] << "  " << info->layer[l].varP << "  " << info->layer[l].kurt << "  " << info->layer[l].check << '\n';
    }

    info->alpha = log2_regression(aveZ);
    info->beta = log2_regression(varZ);
    info->gamma = log2((double)info->layer[test_level].time / (double)info->layer[test_level - 1].time);

    cout << fixed << '\n';
    cout << "alpha = " << info->alpha << '\n';
    cout << "beta  = " << info->beta  << '\n';
    cout << "gamma = " << info->gamma << '\n' << endl;

    ofs << fixed << '\n';
    ofs << "alpha = " << info->alpha << '\n';
    ofs << "beta  = " << info->beta  << '\n';
    ofs << "gamma = " << info->gamma << '\n' << endl;
    ofs.close();
}

void smc_evpi_calc(EvppiInfo *info, int n) {
    double sum_of_max = 0.0;
    vector <double> sum(info->model_num);
    for (int i = 0; i < n; i++) {
        pre_sampling(info);
        post_sampling(info);
        f(info);
        double mx = -1e30;
        for (int j = 0; j < info->model_num; j++) {
            mx = max(mx, info->val[j]);
            sum[j] += info->val[j];
        }
        sum_of_max += mx;
    }

    sum_of_max /= (double)n;
    double max_of_sum = *max_element(sum.begin(), sum.end()) / (double)n;

    cout << scientific << setprecision(3);
    cout << "EVPI = " << sum_of_max - max_of_sum << '\n' << endl;
}

void mlmc_eval_eps(MlmcInfo *info, int level, double eps, vector <int> &n_samples) {
    bool converged = false;
    while (!converged) {
        mlmc_calc(info, level, n_samples);

        for (int l = 2; l <= level; l++) {
            double ave_min = 0.5 * info->layer[l - 1].aveZ / pow(2, info->alpha);
            if (ave_min > info->layer[l].aveZ) {
                info->layer[l].aveZ = ave_min;
            }

            double var_min = 0.5 * info->layer[l - 1].varZ / pow(2, info->beta);
            if (var_min > info->layer[l].varZ) {
                info->layer[l].varZ = var_min;
            }
        }

        double sum = 0.0;
        for (int l = 0; l <= level; l++) {
            sum += sqrt(info->layer[l].varZ * info->layer[l].cost);
        }

        converged = true;
        for (int l = 0; l <= level; l++) {
            n_samples[l] = ceil(max(0.0,
                    sqrt(info->layer[l].varZ / info->layer[l].cost) * sum / ((1.0 - info->theta) * eps * eps) - info->layer[l].n));

            if (n_samples[l] > 0.01 * info->layer[l].n) {
                converged = false;
            }
        }

        if (converged) {
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
                    for (int l = 0; l <= level; l++) {
                        n_samples[l] = ceil(max(0.0,
                                sqrt(info->layer[l].varZ / info->layer[l].cost) * sum / ((1.0 - info->theta) * eps * eps) - info->layer[l].n));
                    }
                }
            }
        }
    }
}

void mlmc_test_eval_eps(MlmcInfo *info, vector <double> &eps, const char *file_name) {
    cout << "eps        value      mlmc       std        save         N...\n";
    cout << "------------------------------------------------------------------\n";

    ofstream ofs(file_name, ios::app);
    ofs << "eps        value      mlmc       std        save         N...\n";
    ofs << "------------------------------------------------------------------\n";

    for (int l = 0; l <= info->max_level; l++) {
        info->layer[l].n = 0;
        info->layer[l].result = result_init();
    }

    int level = 2;
    vector <int> n_samples(info->max_level + 1);
    for (int l = 0; l <= level; l++) n_samples[l] = 1000;

    for (int i = 0; i < (int)eps.size(); i++) {
        mlmc_eval_eps(info, level, eps[i], n_samples);

        for (; level < info->max_level; level++) {
            if (info->layer[level + 1].n == 0) break;
        }

        double value = 0.0;
        for (int l = 0; l <= level; l++) {
            value += info->layer[l].aveZ;
        }

        double mlmc_cost = 0.0;
        for (int l = 0; l <= level; l++) mlmc_cost += info->layer[l].n * info->layer[l].cost;
        double std_cost = info->layer[level].varP * info->layer[level].cost / ((1.0 - info->theta) * eps[i] * eps[i]);

        cout << scientific << setprecision(3) << eps[i] << "  " << value << "  ";
        cout << scientific << setprecision(3) << mlmc_cost << "  " << std_cost << "  ";
        cout << fixed << right << setw(6) << std_cost / mlmc_cost << "  ";

        ofs << scientific << setprecision(3) << eps[i] << "  " << value << "  ";
        ofs << scientific << setprecision(3) << mlmc_cost << "  " << std_cost << "  ";
        ofs << fixed << right << setw(6) << std_cost / mlmc_cost << "  ";

        for (int l = 0; l <= level; l++) {
            cout << fixed << right << setw(9) << info->layer[l].n << ' ';
            ofs << fixed << right << setw(9) << info->layer[l].n << ' ';
        }

        cout << endl;
        ofs << endl;

        fill(n_samples.begin(), n_samples.end(), 0);
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
    result->time = 0;
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
        info->layer[l].time = 0;
        info->layer[l].cost = pow(2.0, l * gamma);
        info->layer[l].aveZ = 0.0;
        info->layer[l].aveP = 0.0;
        info->layer[l].varZ = 0.0;
        info->layer[l].varP = 0.0;
        info->layer[l].kurt = 0.0;
        info->layer[l].check = 0.0;
        info->layer[l].evppi_info = evppi_init(l, m0);
        info->layer[l].result = result_init();
        m0 *= s;
    }
    return info;
}