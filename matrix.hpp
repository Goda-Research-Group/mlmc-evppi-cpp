
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

Matrix operator - (Matrix &A, Matrix &B);
Matrix operator * (Matrix &A, Matrix &B);
vector <double> operator * (vector <double> &v, Matrix &M);
double operator * (vector <double> &v1, vector <double> &v2);
Matrix Cholesky(Matrix &A);
Matrix Inverse(Matrix &A);
void rand_multinormal(vector<double> &u, Matrix &sigma_cholesky, vector<double> &rand, vector<double> &ret);

#endif