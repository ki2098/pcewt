#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace std;

const int P = 5;
const int Division = 64;
const int GuideCell = 2;
const int CellCount = Division + 2*GuideCell;
const double Re = 1000;

double M[P+1][P+1][P+1] = {};

int index(int i, int j, int *size) {
    return i*size[1] + j;
}

int read_M(string &&fname) {
    ifstream m_ifs(fname);
    int i, j, k;
    double m;
    while (!m_ifs.eof()) {
        m_ifs >> i >> j >> k >> m;
        M[i][j][k] = m;
    }
}

double compute_utopia_convection(double ww, double w, double c, double e, double ee, double u, double dx) {
    return (u*(-ee + 8*e - 8*w + ww) + fabs(u)*(ee - 4*e + 6*c - 4*w + ww))/(12*dx);
}

double compute_convection_term(int c, double u[P+1], double v[P+1], double (*val)[P+1], double dx, double dy, int i, int j, int *size) {
    int idc  = index(i, j, size);
    int ide  = index(i + 1, j, size);
    int idee = index(i + 2, j, size);
    int idw  = index(i - 1, j, size);
    int idww = index(i - 2, j, size);
    int idn  = index(i, j + 1, size);
    int idnn = index(i, j + 2, size);
    int ids  = index(i, j - 1, size);
    int idss = index(i, j - 2, size);

    double convection = 0;
    for (int a = 0; a <= P; a ++) {
        for (int b = 0; b <= P; b ++) {
            double m = M[a][b][c];
            double valc  = val[idc ][b];
            double vale  = val[ide ][b];
            double valee = val[idee][b];
            double valw  = val[idw ][b];
            double valww = val[idww][b];
            double valn  = val[idn ][b];
            double valnn = val[idnn][b];
            double vals  = val[ids ][b];
            double valss = val[idss][b];
            double uc = u[a];
            double vc = v[a];
            convection += compute_utopia_convection(valww, valw, valc, vale, valee, uc, dx)*m;
            convection += compute_utopia_convection(valss, vals, valc, valn, valnn, vc, dy)*m;
        }
    }

    return convection;
}
