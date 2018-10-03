#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <armadillo>
#include <stdio.h>
#include "time.h"

#include "JacobiALG.h"

using namespace std;
using namespace arma;


int main(int argc, char *argv[]){
  int n = atoi(argv[1]);
  double omg_r = atof(argv[3]);
  double rho_n = atof(argv[2]);


  double rho_0 = 0.0;
  double h = (rho_n - rho_0)/double(n);

  vec lspcvec = linspace<vec>(1,n,n)*h;
  vec rho = rho_0 + lspcvec;

  //cout << rho << endl;

  //define toepliz matrix for problem
  mat A = q2etoepliz(n, h, omg_r, rho);   //Defines tridiagonal matrix with HO and Coulomb potential

  vec eigval;
  mat eigvec;

  double t_e;
  double t_j;

  clock_t te;
  t_e = clock();

  eig_sym(eigval, eigvec, A);

  te = double (clock() - t_e);

  mat R;
  vec GR;


  clock_t tj;
  t_j = clock();

  mat EJ = jacobi_method(A, R, n, GR);

  tj = double (clock() - t_j);
  cout << "Time using Armadillo: " << te/double(CLOCKS_PER_SEC) << "\n";
  cout << "Time using Jacobi: " << tj/double(CLOCKS_PER_SEC) << endl;

  // writing to file
  ofstream outfile;
  outfile.open("r2e.txt");
  for (int i = 0; i < n; i++) {
    outfile << rho[i] << " ";
    outfile << EJ[i] << " ";
    outfile << GR[i] << " ";
    outfile << eigvec(i,0) << endl;

    }
  outfile.close();
return 0;

}
