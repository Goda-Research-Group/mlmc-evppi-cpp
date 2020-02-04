
#ifndef EVPPI_HPP
#define EVPPI_HPP

#include <vector>

using namespace std;

typedef struct ModelInfo ModelInfo;

typedef struct {
    int level;
    int m; // inner loop の回数
    int model_num;
    vector <double> sample;
    vector <double> val;
    ModelInfo *model_info;
} EvppiInfo;

typedef struct {
    double p1, p2;
    double z1, z2, z3, z4;
    clock_t time;
} Result;

typedef struct {
    int n; // outer loop の回数
    clock_t time;
    double cost;
    double aveZ, aveP, varZ, varP, kurt, check;
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

void sampling_init(EvppiInfo *info);
void pre_sampling(ModelInfo *info);
void post_sampling(ModelInfo *info);
void f(EvppiInfo *info);

void smc_evpi_calc(EvppiInfo *info, int n);

void mlmc_test(MlmcInfo *info, int test_level, int n_sample, const char *file_name = "output.txt");
void mlmc_test_eval_eps(MlmcInfo *info, vector <double> &eps, const char *file_name = "output.txt");
MlmcInfo *mlmc_init(int m0, int s, int max_level, double gamma, double theta);

#endif