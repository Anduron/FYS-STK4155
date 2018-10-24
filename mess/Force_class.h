#ifndef Force_H
#define Force_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <armadillo>
#include <stdio.h>

#include "Planet_class.h"

using namespace std;
using namespace arma;

class Force
{
private:
  double c = 3.0*pow(10,8);

public:

  double A;
  vec<Planet*>PList = {};

  //Default Constructor
  Force(double Acc, cPList);

  //Destructor
  ~Force() {}

  //Constructior with variables
  void Newton(vec M, vec R);


  void Einstein(double A, vec M, vec R);
};

#endif /* Force_H */
