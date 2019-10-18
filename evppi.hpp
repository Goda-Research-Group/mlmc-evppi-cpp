
#ifndef EVPPI_HPP
#define EVPPI_HPP

#include <vector>

using namespace std;

typedef struct {
    int level;
    int m; // inner loop の回数
} EvppiInfo;

typedef struct {
    double dp, dp2;
    double pf, pf2;
} Result;

typedef struct {
    int n; // outer loop の回数
    double cost;
    double aveZ, aveP, varZ, varP;
    EvppiInfo *evppi_info;
    Result *result;
} MlmcLayerInfo;

typedef struct {
    int max_level;
    double alpha;
    double beta;
    double gamma;
    double theta;
    vector <MlmcLayerInfo> layer;
} MlmcInfo;

double f(double x, double y);

void mlmc_test(MlmcInfo *info, int test_level, int n_sample);
void mlmc_test_eval_eps(MlmcInfo *info, vector <double> &eps);
MlmcInfo *mlmc_init(int m0, int s, int max_level, double gamma, double theta);

#endif