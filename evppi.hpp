
#ifndef EVPPI_HPP
#define EVPPI_HPP

#include <vector>

using namespace std;

typedef struct {
    vector <double> x;
} PreInfo;

typedef struct {
    vector <double> y;
} PostInfo;

typedef struct {
    int level;
    int m; // inner loop の回数
    double val;
    PreInfo *pre;
    PostInfo *post;
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

void pre_init(PreInfo *info);
void post_init(PostInfo *info);
void pre_sampling(PreInfo *info);
void post_sampling(PostInfo *info);

double f1(EvppiInfo *info);
double f2(EvppiInfo *info);

void mlmc_test(MlmcInfo *info, int test_level, int n_sample, const char *file_name = "output.txt");
void mlmc_test_eval_eps(MlmcInfo *info, vector <double> &eps, const char *file_name = "output.txt");
MlmcInfo *mlmc_init(int m0, int s, int max_level, double gamma, double theta);

#endif