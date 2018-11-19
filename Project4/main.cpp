#include <fstream>
#include <iomanip>
#include <string>
#include <iostream>
#include <cmath>
#include <random>
#include <map>
#include <armadillo>
#include <stdio.h>


#include "mpi.h"
#include "time.h"

using namespace std;
using namespace arma;

ofstream outfile;

int p_bc(int i, int bound, int n_add){
  return (i+bound+n_add)%(bound);
}

void Energy_lattice(int n_Spin, int GS, mat &spin_mat, double &Enr, double &Mag, mt19937_64 &generator)
{
  if (GS == 0){
    spin_mat.fill(1.0);
    Mag += (double) n_Spin * (double) n_Spin;
  }
  else{
    uniform_int_distribution<int> rngSpin(0,1);
    for(int i = 0; i < n_Spin; i++){
      for(int j = 0; j < n_Spin; j++){
        spin_mat(i,j) = 2*rngSpin(generator) - 1;
      }
    }
  }

  for(int i = 0; i < n_Spin; i++){
    for(int j = 0; j < n_Spin; j++){
      Enr -= (double) spin_mat(i,j)*
      (spin_mat(p_bc(i, n_Spin, -1), j) +
      spin_mat(i ,p_bc(j, n_Spin, -1)));

      Mag += (double) spin_mat(i,j);
    }
  }
}

map<double , double> Flip(double Tmp)
{
  map<double , double> accF;

  for (int dE = -8; dE <= 8; dE+=4){
    accF.insert(pair<double , double>(dE,exp(-dE/Tmp)));
  }
  return accF;
}

void metropolis_alg(int n_Spin, int s_MC, int n_MC, double Tmp, vec &Exp_Val, vec &Tot_vals, vec &Enr_vec, vec &Mag_vec, map<double,double>accF, int cutoff)
{
  random_device rd;
  mt19937_64 generator(rd());
  uniform_real_distribution<double> rng(0.0,1.0);
  uniform_int_distribution<int> rngPos(0,n_Spin-1);

  mat spin_mat = zeros<mat>(n_Spin,n_Spin);
  double Enr = 0.0; double Mag = 0.0;

  Energy_lattice(n_Spin, 1, spin_mat, Enr, Mag, generator);

  int a_counter = 0;

  for (int cycles = s_MC; cycles <= n_MC; cycles ++){
    for (int i = 0; i < n_Spin; i++){
      for (int j = 0; j < n_Spin; j++){

        int idx_i = rngPos(generator);

        int idx_j = rngPos(generator);

        int dE = 2.0*spin_mat(idx_i,idx_j)*
        ( spin_mat( idx_i, p_bc( idx_j, n_Spin, -1) )
        + spin_mat( p_bc(idx_i, n_Spin ,-1 ), idx_j )
        + spin_mat( idx_i, p_bc( idx_j, n_Spin, 1) )
        + spin_mat( p_bc(idx_i ,n_Spin ,1 ), idx_j) );


        if (rng(generator) <= accF.find(dE)->second){
          spin_mat(idx_i,idx_j) *= -1.0;
          Mag += (double) 2*spin_mat(idx_i,idx_j);
          Enr += (double) dE;
          a_counter += 1;
        }
      }
    }
    //outfile << a_counter << endl;
    //Enr_vec(cycles) = Enr/(n_Spin*n_Spin);
    //cout << Enr_vec(cycles);

    if (cycles > s_MC + cutoff){
      Enr_vec(cycles) = Enr/(n_Spin*n_Spin);
      Mag_vec(cycles) = Mag/(n_Spin*n_Spin);
      Exp_Val(0) += Enr; Exp_Val(1) += Enr*Enr;
      Exp_Val(2) += Mag; Exp_Val(3) += Mag*Mag; Exp_Val(4) += fabs(Mag);
    }
  }
  for (int exn = 0; exn < 5; exn++){
    MPI_Reduce(&Exp_Val(exn), &Tot_vals(exn), 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  cout << a_counter << endl;
}

void Results(int n_Spin, int n_MC, double Tmp, vec &Tot_vals, int numprocs, int cutoff){
  double n_factor = 1.0/((double) (n_MC-cutoff));
  double E_Exp_Val = Tot_vals(0)*n_factor;
  double EE_Exp_Val = Tot_vals(1)*n_factor;
  double M_Exp_Val = Tot_vals(2)*n_factor;
  double MM_Exp_Val = Tot_vals(3
  )*n_factor;
  double absM_Exp_Val = Tot_vals(4)*n_factor;

  double Energy_variance = (EE_Exp_Val - E_Exp_Val * E_Exp_Val )/ (n_Spin*n_Spin);
  double MagMoment_variance = (MM_Exp_Val - absM_Exp_Val*absM_Exp_Val)/ (n_Spin*n_Spin);

  if (numprocs == 1){
    double MagMoment_variance = (MM_Exp_Val)/ (n_Spin*n_Spin);
  }

  outfile << Tmp << " ";
  outfile << E_Exp_Val /( n_Spin * n_Spin ) << " ";
  outfile << EE_Exp_Val / ( n_Spin * n_Spin ) << " ";
  outfile << M_Exp_Val / ( n_Spin * n_Spin ) << " ";
  outfile << MM_Exp_Val / ( n_Spin * n_Spin ) << " ";
  outfile << absM_Exp_Val / ( n_Spin * n_Spin ) << " ";
  outfile << Energy_variance << " ";
  outfile << MagMoment_variance << endl;


  cout << Tmp << " ";
  cout << E_Exp_Val/(n_Spin*n_Spin) << " ";
  cout << EE_Exp_Val/(n_Spin*n_Spin) << " ";
  cout << M_Exp_Val/(n_Spin*n_Spin) << " ";
  cout << MM_Exp_Val/(n_Spin*n_Spin) << " ";
  cout << absM_Exp_Val/(n_Spin*n_Spin) << endl;
}


int main(int argc, char* argv[])
{
int n_Spin, n_MC, my_rank, numprocs, cutoff;
double T_s, T_f, T_dt, temp;
MPI_Status status;


n_Spin = 20;
n_MC = 1000000;
cutoff = 0;
temp = 2.4;

//T_s = 2.2;
//T_f = 2.4;
//T_dt = 0.01;

outfile.open("r4b.txt");


vec Exp_Val = zeros<vec>(5);
vec Tot_vals = zeros<vec>(5);
vec Enr_vec = zeros<vec>(n_MC+1);
vec Mag_vec = zeros<vec>(n_MC+1);

//MPI, parallelization
MPI_Init (&argc, &argv);
MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);

int intervalls = n_MC/numprocs;
int beginloop = my_rank*intervalls + 1;
int endloop = (my_rank+1)*intervalls;
if ( (my_rank == numprocs - 1) && (endloop < n_MC)) endloop = n_MC;

//MPI_Bcast (&n_Spin, 1, MPI_INT, 0, MPI_COMM_WORLD);
//MPI_Bcast (&T_s, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//MPI_Bcast (&T_f, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//MPI_Bcast (&T_dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);



double Tstart, Tend;
Tstart = MPI_Wtime();

map <double,double> accF = Flip(temp);
metropolis_alg(n_Spin,0,n_MC,temp,Exp_Val,Tot_vals,Enr_vec,Mag_vec,accF,cutoff);
Results(n_Spin,n_MC,temp,Exp_Val, numprocs,cutoff);

for (int k = 1; k <= n_MC ;k++){
  outfile << Enr_vec(k) << " " << Mag_vec(k) << endl;
}
//for(double temp = T_s; temp <= T_f; temp += T_dt){
//  vec Exp_Val = zeros<vec>(5);
//  map <double,double> accF = Flip(temp);
//  metropolis_alg(n_Spin,beginloop,endloop,temp,Exp_Val,Tot_vals,Enr_vec,accF,cutoff);
//  if (my_rank == 0){
//    Results(n_Spin,endloop,temp,Exp_Val, numprocs,cutoff);
//  }

//}

Tend = MPI_Wtime();

if (my_rank == 0){
  cout << "time" << " " << Tend - Tstart << endl;
}
outfile.close();

MPI_Finalize ();
}
