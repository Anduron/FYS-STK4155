#ifndef Planet_H
#define Planet_H

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

class Planet
{
public:

  vec pos;
  vec vel;
  double M;

  //Default Constructor
  Planet();

  //Constructior with variables
  Planet(Position, Velocity, Mass);


  //Destructor
  ~Planet() {}

};
