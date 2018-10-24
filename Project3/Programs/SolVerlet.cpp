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

int main(int argc, char *argv[]){
  double T = atoi(argv[1]);
  double n = 10000*T;
  double t_0;

  /*
  if( T>100 ){
    n = 1000*T;
  }
  else{
    n = 1000*T;
  }
  */
  vec a = zeros(n);
  vec vx = zeros(n), vy = zeros(n), vz = zeros(n);
  vec x = zeros(n), y = zeros(n), z = zeros(n);
  double r;
  x(0) = 1, y(0) = 0, z(0) = 0,  vy(0) = 2*M_PI;
  double m_O = 2*pow(10,30), m_E = 6*pow(10,24)/m_O;
  double G = 4*M_PI*M_PI;

  double dt = T/n;


  clock_t t;
  t_0 = clock();

  for (int i = 0; i < n-1; i+=1){
    r = sqrt(x(i)*x(i) + y(i)*y(i) + z(i)*z(i));
    a(i) = -G/pow(r,3);
    vx(i+1) = vx(i) + a(i)*x(i)*dt;
    vy(i+1) = vy(i) + a(i)*y(i)*dt;
    vz(i+1) = vz(i) + a(i)*z(i)*dt;

    x(i+1) = x(i) + vx(i)*dt;
    y(i+1) = y(i) + vy(i)*dt;
    z(i+1) = z(i) + vz(i)*dt;

  }
  //rewrite as verlet!!!!!!!!!!!!!!!!!!!!!!!!!!
  /*
  for (int i = 0; i < n-1; i++){

    x(i+1) = x(i) + dt*vx(i) + dt*(dt/2)*a(i)*x(i);
    y(i+1) = y(i) + dt*vy(i) + dt*(dt/2)*a(i)*y(i);
    z(i+1) = z(i) + dt*vz(i) + dt*(dt/2)*a(i)*z(i);

    r = sqrt(x(i+1)*x(i+1) + y(i+1)*y(i+1) + z(i+1)*z(i+1));
    a(i+1) = -(G)/pow(r,3);

    vx(i+1) = vx(i) + (dt/2)*(a(i+1)*x(i+1)+a(i)*x(i));
    vy(i+1) = vy(i) + (dt/2)*(a(i+1)*y(i+1)+a(i)*y(i));
    vz(i+1) = vz(i) + (dt/2)*(a(i+1)*z(i+1)+a(i)*z(i));
  }
  */

  t = double (clock() - t_0);

  cout << "Time using Euler: " << t/double(CLOCKS_PER_SEC) << "\n";

  // writing to file
  ofstream outfile;
  outfile.open("r3a.txt");
  for (int i = 0; i < n-1; i+=10) {         //Prints results to file
    outfile << x[i] << " ";
    outfile << y[i] << " ";
    outfile << z[i] << endl;
  }
  outfile.close();
}
