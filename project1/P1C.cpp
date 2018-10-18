#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <armadillo>
#include <stdio.h>
#include "time.h"

using namespace std;

// object for output files
ofstream ofile;
// Functions used
inline double f(double x){return 100.0*exp(-10.0*x);
}
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


  double h = 1.0/(n+1);

  double t0;
  double *u = new double[n+2];
  double *bt = new double[n+2];
  double *ft = new double[n+2];
  double *fprime = new double[n+2];
  double *x = new double[n+2];
  double flop;
  //the a, b and c is now known
  //bt is b-tilde and ft is f-tilde, u is the vector

  clock_t t;

  t0 = clock();

  double a = -1; double b = 2; double c = -1;

  for (int i = 0; i <n+2; i++){
    x[i] = h*double(i);
    fprime[i] = h*h*f(x[i]);
  }

  //initial conditions
  ft[0] = fprime[0]; ft[1] = fprime[1];
  u[0] = 0; u[n+1] = 0;
  bt[0] = b; bt[1] = b;


  //forward substitution
  for (int i = 2; i < n+2; i++){
    flop = (a/bt[i-1]);
    bt[i] = b - flop*c;
    ft[i] = fprime[i] - flop*ft[i-1];
  }

  //bacward substitution
  for (int i = n; i > 0; i--){
    u[i] = (ft[i] - c*u[i+1])/bt[i];
  }

  t = double (clock() - t0);

  ofile.open(filename);
  cout << "Time to run algorithm: " << t/double(CLOCKS_PER_SEC) << "\n";

  for(int i=0; i < n+2; i++) {
               ofile << x[i];
               ofile << " " << u[i];
               ofile << " " << exact(x[i]) << endl;
  }
  ofile.close();

  return 0;
}
