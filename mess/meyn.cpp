#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <armadillo>
#include <stdio.h>
#include "time.h"

#include "Force_class.h"
#include "Planet_class.h"

using namespace std;
using namespace arma;


int main(int argc, char *argv[]){
  double T = atoi(argv[1]);
  double n = 1000*T;
  double m_E = 6*pow(10,24)/(2*pow(10,30));
  vec Epos = {1,0,0};
  vec Evel = {0,2*M_PI,0};

  Planet Earth(Epos,Evel,m_E);
};
