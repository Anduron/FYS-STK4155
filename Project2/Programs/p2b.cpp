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


  double rho_0 = 0.0;
  double rho_n = 1.0;
  double h = (rho_n - rho_0)/double(n); //Step size based on interval rho


  mat A = toepliz(n, h);                //Defines the toepliz matrix

  vec eigval;
  mat eigvec;

  double t_e;
  double t_j;

  clock_t te;
  t_e = clock();

  eig_sym(eigval, eigvec, A);           //Runs armadillo functions

  te = double (clock() - t_e);          //Times armadillo

  mat R;
  vec GR;


  clock_t tj;
  t_j = clock();

  mat EJ = jacobi_method(A, R, n, GR);  //Runs the jacobi method

  tj = double (clock() - t_j);          //Times Jacobi's method

  cout << "Time using Armadillo: " << te/double(CLOCKS_PER_SEC) << "\n";
  cout << "Time using Jacobi: " << tj/double(CLOCKS_PER_SEC) << endl;
  vec eigenvals = eigvals_analytical(n, h);
  // writing to file
  ofstream outfile;
  outfile.open("r2b.txt");
  for (int i = 0; i < n; i++) {         //Prints results to file
    outfile << EJ[i] << " ";
    outfile << eigval[i] << " ";
    outfile << eigenvals[i] << endl;
    printf("EJ = %1.8g, eig = %1.8g \n", EJ[i], eigenvals[i]);
    }
  outfile.close();

return 0;

}
