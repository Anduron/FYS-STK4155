#ifndef JALG_H
#define JALG_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <armadillo>
#include <stdio.h>
#include "time.h"

using namespace std;
using namespace arma;

double maxoffdiag (mat &A, int &k, int &l, int n);
void rotation (mat &A, mat &R, int &k, int &l, int n);
vec jacobi_method (mat &A ,mat &R ,int n, vec &GR);
vec eigvals_analytical(int n, double h); 

mat toepliz (int n, double h);
mat qtoepliz (int n, double h, vec &rho);
mat q2etoepliz (int n, double h, double &omg_r, vec &rho);

#endif /* JALG_H */
