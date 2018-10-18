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
ofstream ofile;

inline double f(double x){return 100.0*exp(-10.0*x);}
inline double exact(double x) {return 1.0-(1-exp(-10))*x-exp(-10*x);}

int main(int argc, char *argv[]){
  int n;
  string filename;
  if( argc <= 1 ){
        cout << "Need name and value of n" << endl;
        exit(1);
  }
  else{
      filename = argv[1];
      n = atoi(argv[2]);
  }

  double t0;
  clock_t t;
  t0 = clock();

  double h = 1.0/(n+1);
  mat A = zeros<mat>(n+2,n+2);
  A.diag() += 2.0;
  A.diag(1) -= 1.0;
  A.diag(-1) -= 1.0;

  double *fprime = new double[n+2];
  vec ft = zeros(n+2);
  vec u = zeros(n+2);


  double* x = new double[n+2];

  for (int i = 1; i <n+2; i++){
    x[i] = h*double(i);
    ft[i] = h*h*f(x[i]);
  }


  u = solve(A,ft);

  t = double (clock() - t0);

  cout << "Time to run algorithm: " << t/double(CLOCKS_PER_SEC) << "\n";

  ofile.open(filename);

  for(int i=0; i < n+2; i++){
               ofile << x[i];
               ofile << " " << u(i);
               ofile << " " << exact(x[i]) << endl;
  }
  ofile.close();


  return 0;
}
