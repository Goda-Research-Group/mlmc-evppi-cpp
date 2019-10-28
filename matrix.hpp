
#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <random>

using namespace std;

struct Matrix {
    vector < vector<double> > val;
    Matrix(int n, int m) { val.clear(); val.resize(n, vector<double>(m)); }
    int size() { return val.size(); }
    inline vector<double> &operator[](int i) { return val[i]; }
};

Matrix Cholesky(Matrix &A);
vector<double> rand_multinormal(vector<double> &u, Matrix &sigma_cholesky);

#endif