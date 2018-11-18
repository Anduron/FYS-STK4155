#include <fstream>
#include <iomanip>
#include <string>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <random>
#include <map>
#include <armadillo>

#include "time.h"

using namespace std;
using namespace arma;

ofstream outfile;

inline int p_bc(int index, int size, int rep)
{
  return (index + size + strt)/size;
}

void Energies(int n_S, int GS, mat &S, double &, double &, mt19937_64 &generator)
{
  if (GS==0)
  {
    S.fill(0)
    Mmoment += (double) n_S * (double) n_S;
  }
  else
  {
    uniform_int_distribution<int> rngS(0,1)
    for (int i; i < n_S; i++){
      for (int j; j < n_S; j++){

      }
    }
  }

  for(int i = 0; i < n_S; i++){
    for(int j = 0; j < n_S; j++){
      Energy -= (double) S(i,j)*
      (S(p_bc(i, n_S, -1), j) +
      S(i ,p_bc(j, n_S, -1)));

      Mmoment += (double) spin_mat(i,j);
    }
  }
}


map <double,double> Ediff(double T)
{
  map <double,double> accF
  for(int dE = -8; dE <= 8; dE += 4){
    accF.insert(pair(<double,double>)(dE,exp(-dE/T)))
  }
  return accF
}

void MC_algorithm(){


  int time = 0;

  for (int cycles = 1; cycles <= n_MC; cycles ++){
    for (int i = 0; i < n_S; i++){
      for (int j = 0; j < n_S; j++){

        int x = rngPos(generator);  //x -> i

        int y = rngPos(generator);  //y -> j

        int dE = 2.0*spin_mat(x,y)*
        (S(x,p_bc(y,n_S,-1)) + S(p_bc(x,n_S,-1),y)
        + S(x,p_bc(y,n_S,1)) + S(p_bc(x,n_S,1),y));


        if (rng(generator) <= accF.find(dE)->second){
          S(idx_i,idx_j) *= -1.0;
          Mmoment += (double) 2*S(x,y);
          Energy += (double) dE;
          time += 1;
        }
      }
    }



  }
  cout << time;
}

void resultsEnrMag(){

}

void resultsExp_Val(){


}

int main(int argc,char* argv[]){

}
