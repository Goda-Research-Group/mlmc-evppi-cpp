
#include <iostream>
#include <iomanip>
#include "matrix.hpp"

using namespace std;

random_device rd;
mt19937 generator(rd());
normal_distribution<double> dist(0.0, 1.0);

int main() {
    double rho = 0.6;
    vector <double> u = {0.7, 3.0, 0.8, 3.0};
    vector <double> x = {0.1, 0.5, 0.1, 1.0};
    Matrix sigma(4, 4);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            sigma[i][j] = x[i] * x[j];
            if (i != j) sigma[i][j] = rho * x[i] * x[j];
        }
    }
    Matrix sigma_cholesky = Cholesky(sigma);

    int n = 1000000;
    vector<vector<double>> val(n, vector<double>(4));
    for (int i = 0; i < n; i++) {
        vector <double> ran(4);
        for (int j = 0; j < 4; j++) ran[j] = dist(generator);
        vector <double> r = rand_multinormal(u, sigma_cholesky, ran);
        for (int j = 0; j < 4; j++) val[i][j] = r[j];
    }

    cout << scientific << setprecision(4);

    cout << "average =\n";
    vector <double> ave(4);
    for (int j = 0; j < 4; j++) {
        double sum = 0;
        for (int i = 0; i < n; i++) sum += val[i][j];
        ave[j] = sum / n;
        cout << ave[j] << ' ';
    }
    cout << "\n\n";

    cout << "standard derivation =\n";
    vector <double> s(4);
    for (int j = 0; j < 4; j++) {
        double sum = 0;
        for (int i = 0; i < n; i++) sum += (val[i][j] - ave[j]) * (val[i][j] - ave[j]);
        s[j] = sum / n;
        cout << sqrt(s[j]) << ' ';
    }
    cout << "\n\n";

    cout << "col matrix =\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) sum += (val[k][i] - ave[i]) * (val[k][j] - ave[j]);
            cout << sum / n / sqrt(s[i] * s[j]) << ' ';
        }
        cout << '\n';
    }

    cout << endl;
    return 0;
}