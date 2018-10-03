#include "JacobiALG.h"

mat toepliz (int n, double h){
  double a = -1.0/(h*h);      //Upper and lower diagonal
  double d = 2.0/(h*h);       //Diagonal
  mat A = zeros<mat>(n, n);
  A.diag() += d;
  A.diag(1) += a;             //Upper diagonal
  A.diag(-1) += a;            //Lower diagonal
  return A;
}

mat qtoepliz (int n, double h, vec &rho){
  //No longer a toeplitz matrix
  double a = -1.0/(h*h);
  double d = 2.0/(h*h);
  mat A = zeros<mat>(n, n);
  A.diag() += d + rho%rho;    //Diagonal with Harmonic oscillator potenital
  A.diag(1) += a;
  A.diag(-1) += a;
  return A;
}

mat q2etoepliz (int n, double h, double &omg_r, vec &rho){
  //No longer a toeplitz matrix
  double a = -1.0/(h*h);
  double d = 2.0/(h*h);
  mat A = zeros<mat>(n, n);
  A.diag() += d + omg_r*omg_r*rho%rho + (1.0/rho); //Diagonal with HO potential and Coulomb interaction
  A.diag(1) += a;
  A.diag(-1) += a;
  return A;
}
