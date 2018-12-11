#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <armadillo>
#include <stdio.h>
#include "time.h"

using namespace std;



void ForwardEuler(int n, double time, double dx, double k){

}

void BackwardEuler(int n, double time, double dx, double k){
  vec u = zeros<vec>(n);
  vec f = zeros<vec>(n);

  for (int t = 0; t = time; t++){

    gaussElim(t, a, b, c, f, u)


  }
}

void CrankNicholson(int n, double time, double dx, double k){

}

void gaussElim(int t, double d0, double d1, vec f, vec u){
  double temp

  for (int i = 0; i < t; i++){

    ft(i) = f(i) - ft(i-1)*temp
  }
}

int main(int argc, char *argv[]){

}
