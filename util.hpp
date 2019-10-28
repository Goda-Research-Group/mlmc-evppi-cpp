
#ifndef UTIL_HPP
#define UTIL_HPP

using namespace std;

double expit(double x);
double logit(double x);
double regression(vector <double> &x, vector <double> &y);
double log2_regression(vector <double> &y);

#endif