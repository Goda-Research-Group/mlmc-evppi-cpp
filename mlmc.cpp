#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <math.h>
#include <iomanip>
#include "matplotlibcpp.h"
#define double long double
using namespace std;
namespace plt = matplotlibcpp;
typedef pair<double, double> Pair;

const double zero = 0.0;
const int Lmax = 30;

vector<double> N(Lmax);
vector<double> dN(Lmax);
vector<double> P(Lmax);
vector<double> P2(Lmax);
vector<double> aveP(Lmax);
vector<double> varP(Lmax);
vector<double> Z(Lmax);
vector<double> Z2(Lmax);
vector<double> aveZ(Lmax);
vector<double> varZ(Lmax);
vector<double> Cost(Lmax);
vector<double> Kurtosis(Lmax);

// It constructs a trivial random generator engine from a time-based seed
unsigned seed = chrono::system_clock::now().time_since_epoch().count();
default_random_engine generator(seed);

// Initializes the normal distribution
normal_distribution<double> distribution(zero, 1.0);

double f1(double x, double y) {
    return x + y;
}

double f2(double x, double y) {
    return x * x * x + y;
}

double f3(double x, double y) {
    if (x < -1) {
        return x + 1 + y;
    } else if (x > 1) {
        return x - 1 + y;
    } else {
        return y;
    }
}

double regression(vector <double> x, vector <double> y) {
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

void mlmc_calc(int l, double M) {
    double z3= 0.0, z4 = 0.0;

    for (int n = 0; n < dN[l]; n++) {
        double x = distribution(generator);

        double max_sum = 0.0, sum_max = 0.0;
        for (int m = 0; m < M; m++) {
            double y = distribution(generator);
            double f = f3(x, y);

            max_sum += max(zero, f);
            sum_max += f;
        }

        max_sum /= M;
        sum_max /= M;
        sum_max = max(zero, sum_max);

        double p = max_sum - sum_max;
        P[l] += p;
        P2[l] += p * p;
    }

    if (l) {
        for (int n = 0; n < dN[l]; n++) {
            double x = distribution(generator);

            double max_sum = 0.0, sum_max = 0.0;
            double first_max_sum = 0.0, first_sum_max = 0.0;
            double second_max_sum = 0.0, second_sum_max = 0.0;

            for (int m = 0; m < M / 2; m++) {
                double y = distribution(generator);
                double f = f3(x, y);

                first_max_sum += max(zero, f);
                max_sum += max(zero, f);

                first_sum_max += f;
                sum_max += f;
            }

            for (int m = M / 2; m < M; m++) {
                double y = distribution(generator);
                double f = f3(x, y);

                second_max_sum += max(zero, f);
                max_sum += max(zero, f);

                second_sum_max += f;
                sum_max += f;
            }

            max_sum /= M;
            first_max_sum /= (M / 2);
            second_max_sum /= (M / 2);

            sum_max /= M;
            first_sum_max /= (M / 2);
            second_sum_max /= (M / 2);

            sum_max = max(zero, sum_max);
            first_sum_max = max(zero, first_sum_max);
            second_sum_max = max(zero, second_sum_max);

            double z = max_sum - sum_max;
            z -= (first_max_sum - first_sum_max) / 2;
            z -= (second_max_sum - second_sum_max) / 2;

            Z[l] += z;
            Z2[l] += z * z;
            z3 += z * z * z;
            z4 += z * z * z * z;
        }
    }

    aveZ[l] = Z[l] / N[l];
    aveP[l] = P[l] / N[l];

    varZ[l] = Z2[l] / N[l] - aveZ[l] * aveZ[l];
    varP[l] = P2[l] / N[l] - aveP[l] * aveP[l];

    Kurtosis[l] = 0.0;
    Kurtosis[l] += z4 / N[l];
    Kurtosis[l] -= 4 * aveZ[l] * z3 / N[l];
    Kurtosis[l] += 6 * aveZ[l] * aveZ[l] * Z2[l] / N[l];
    Kurtosis[l] -= 3 * aveZ[l] * aveZ[l] * aveZ[l] * aveZ[l];
    Kurtosis[l] /= (varZ[l] * varZ[l]);
}

Pair mlmc_estimate_alpha_beta(int L, double n, double M) {
    fill(N.begin(), N.end(), n);
    fill(dN.begin(), dN.end(), n);
    fill(P.begin(), P.end(), 0.0);
    fill(P2.begin(), P2.end(), 0.0);
    fill(Z.begin(), Z.end(), 0.0);
    fill(Z2.begin(), Z2.end(), 0.0);

    cout << " l    aveZ          aveP          varZ          varP\n";
    cout << "----------------------------------------------------------\n";

    for (int l = 0; l <= L; l++, M *= 2) {
        mlmc_calc(l, M);

        cout << setw(2) << right << l << "    ";
        cout << setprecision(5) << scientific;
        cout << aveZ[l] << "   " << aveP[l] << "   " << varZ[l] << "   " << varP[l] << '\n';
    }

    vector<double> x(L);
    vector<double> y(L);

    // liner regression for alpha
    for (int l = 1; l <= L; l++) {
        x[l - 1] = l;
        y[l - 1] = -log2(aveZ[l]);
    }
    double alpha = regression(x, y);
    cout << "alpha = " << fixed << alpha << "\n";

    // liner regression for beta
    for (int l = 1; l <= L; l++) {
        x[l - 1] = l;
        y[l - 1] = -log2(varZ[l]);
    }
    double beta = regression(x, y);
    cout << "beta =  " << fixed << beta << "\n\n";

    return Pair(alpha, beta);
}

bool mlmc_eval_eps(int L, double eps, double alpha, double beta) {
    fill(N.begin(), N.end(), 0);
    fill(dN.begin(), dN.end(), 0);
    fill(P.begin(), P.end(), zero);
    fill(P2.begin(), P2.end(), zero);
    fill(aveP.begin(), aveP.end(), zero);
    fill(varP.begin(), varP.end(), zero);
    fill(Z.begin(), Z.end(), zero);
    fill(Z2.begin(), Z2.end(), zero);
    fill(aveZ.begin(), aveZ.end(), zero);
    fill(varZ.begin(), varZ.end(), zero);
    fill(Cost.begin(), Cost.end(), zero);

    int n0 = 1000;
    double gamma = 1.0;
    double theta = 0.25;

    for (int l = 0; l <= L; l++) dN[l] = n0;
    for (int l = 0; l <= L; l++) Cost[l] = pow(2.0, l * gamma);

    bool converged = false;
    while (!converged) {
        for (int l = 0; l <= L; l++) N[l] += dN[l];

        double M = 1;
        double sum = 0.0;
        for (int l = 0; l <= L; l++, M *= 2) {
            mlmc_calc(l, M);
            sum += sqrt(varZ[l] * Cost[l]);
        }

        double diff = 0;
        for (int l = 0; l <= L; l++) {
            dN[l] = ceil(max(zero, sqrt(varZ[l] / Cost[l]) * sum / ((1.0 - theta) * eps * eps) - N[l]));
            diff += max(zero, dN[l] - 0.01 * N[l]);
        }

        if (diff == 0) {
            converged = true;
            double rem = aveZ[L] / (pow(2.0, alpha) - 1.0);

            if (rem > sqrt(theta) * eps) {
                if (L == Lmax - 1) {
                    cout << " Level over !!!\n";
                    return false;
                } else {
                    converged = false;
                    L++;
                    varZ[L] = varZ[L - 1] / pow(2.0, beta);
                    Cost[L] = Cost[L - 1] * pow(2.0, gamma);

                    sum += varZ[L] * Cost[L];
                    for (int l = 0; l <= L; l++) {
                        dN[l] = ceil(max(zero, sqrt(varZ[l] / Cost[l]) * sum / ((1.0 - theta) * eps * eps) - N[l]));
                    }
                }
            }
        }
    }

    double M = 1;
    double mlmc_cost = 0;
    for (int l = 0; l <= L; l++, M *= 2) mlmc_cost += N[l] * M;

    double std_cost = varP[L] * (M / 2) / ((1.0 - theta) * eps * eps);

    cout << setprecision(3) << scientific;
    cout << eps << "   " << mlmc_cost << "   " << std_cost << "   " << std_cost / mlmc_cost << "    ";
    for (int i = 0; i <= L; i++) cout << N[i] << " ";
    cout << "\n";

    return true;
}

template <typename T>
vector <T> vector_slice(vector <T> vec, int l, int r, int mode = 1) {
    vector <T> ret;
    if (mode == 1) {
        for (int i = l; i <= r; i++) ret.push_back(vec[i]);
    } else if (mode == 2) {
        for (int i = l; i <= r; i++) ret.push_back(log2(vec[i]));
    } else if (mode == 10) {
        for (int i = l; i <= r; i++) ret.push_back(log10(vec[i]));
    }

    return ret;
}

int main() {
    double alpha = -1;
    double beta = -1;

    vector <int> level(Lmax);
    for (int l = 0; l < Lmax; l++) level[l] = l;

    if (alpha < 0 || beta < 0) {
        int sz = 10;
        Pair param = mlmc_estimate_alpha_beta(sz, 200000, 1);

        plt::title("log2 variance / level");
        plt::plot(vector_slice(level, 1, sz), vector_slice(varP, 1, sz, 2), "r^-");
        plt::plot(vector_slice(level, 1, sz), vector_slice(varZ, 1, sz, 2), "b^-");
        plt::xlim(0, sz);
        plt::show();

        plt::title("log2 |mean| / Level");
        plt::plot(vector_slice(level, 1, sz), vector_slice(aveP, 1, sz, 2), "rs-");
        plt::plot(vector_slice(level, 1, sz), vector_slice(aveZ, 1, sz, 2), "bs-");
        plt::xlim(0, sz);
        plt::show();

        plt::title("kurtosis");
        plt::plot(vector_slice(level, 1, sz), vector_slice(Kurtosis, 1, sz), "rs-");
        plt::xlim(0, sz);
        plt::show();

        alpha = param.first;
        beta = param.second;
    }

    double eps[6] = {0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001};
    string graphic[6] = {"b^-", "g^-", "r^-", "bs-", "gs-", "rs-"};

    cout << "eps         mlmc_cost   std_cost    saving       N...\n";
    cout << "----------------------------------------------------------------\n";
    plt::title("N / Level");

    vector <double> std_cost(6);
    vector <double> mlmc_cost(6);

    double theta = 0.25;

    for (int i = 0; i < 6; i++) {
        bool flag = mlmc_eval_eps(2, eps[i], alpha, beta);
        if (!flag) return 0;

        int sz = 0;
        for (; sz < Lmax; sz++) if (N[sz] <= 0) break;
        sz--;

        plt::plot(vector_slice(level, 1, sz), vector_slice(N, 1, sz, 10), graphic[i]);

        double TotalCost = 0.0;
        for (int l = 0; l <= sz; l++) TotalCost += N[l] * Cost[l];
        mlmc_cost[i] = eps[i] * eps[i] * TotalCost;
        std_cost[i] = varP[sz] * Cost[sz] / (1.0 - theta);
    }


    plt::show();

    plt::title("Cost / accuracy");
    plt::plot(vector_slice(vector <double> (eps, eps + 6), 0, 5, 10), vector_slice(mlmc_cost, 0, 5, 10), "r^-");
    plt::plot(vector_slice(vector <double> (eps, eps + 6), 0, 5, 10), vector_slice(std_cost, 0, 5, 10), "b^-");
    plt::show();

    return 0;
}