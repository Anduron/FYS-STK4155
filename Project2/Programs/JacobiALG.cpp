#include "JacobiALG.h"


double maxoffdiag (mat &A, int &k, int &l, int n){
  double maximum = 0.0;

  for (int i = 0; i < n; i++){

    for (int j = i + 1; j < n; j++){

      if (fabs(A(i,j))>maximum){      //Tests all elemtens and sets maximum to largest
          maximum = fabs(A(i,j));     //Sets the maximum element
          l = i;                      //Sets the next index for rotation
          k = j;
      }
    }
  }
  return maximum;
}


void rotation (mat &A, mat &R, int &k, int &l, int n){
  double c, s;

  if (A(k,l) != 0.0) {
    double t, tau;
    tau = (A(l,l) - A(k,k))/(2*A(k,l));

    if (tau > 0) {
      t = 1.0/(tau + sqrt(1.0 + tau*tau));
    } else {
      t = -1.0/(-tau + sqrt(1.0 + tau*tau));
    }

    c = 1/sqrt(1 + t*t);
    s = c*t;

  } else {
    c = 1.0;
    s = 0.0;
  }

  double a_kk, a_ll, a_ik, a_il, r_ik, r_il;
  a_kk = A(k,k);
  a_ll = A(l,l);


  A(k,k) = c*c*a_kk - 2.0*c*s*A(k,l) + s*s*a_ll;
  A(l,l) = s*s*a_kk + 2.0*c*s*A(k,l) + c*c*a_ll;
  A(k,l) = 0.0;
  A(l,k) = 0.0;

  for (int i = 0; i < n; i++) {
    if (i != k && i != l) {
      a_ik = A(i,k);
      a_il = A(i,l);
      A(i,k) = c*a_ik - s*a_il;
      A(k,i) = A(i,k);
      A(i,l) = c*a_il + s*a_ik;
      A(l,i) = A(i,l);
    }

    r_ik = R(i,k);
    r_il = R(i,l);
    R(i,k) = c*r_ik - s*r_il;
    R(i,l) = c*r_il + s*r_ik;

  }
  return;
}



vec jacobi_method (mat &A ,mat &R ,int n, vec &GR){
  int k, l;
  double eps = 1.0e-8;                              //prefixed test
  double max_itr = double(n)*double(n)*double(n);   //maximum number of iterations
  double max_offdiag = maxoffdiag(A,k,l,n);         //maximum size of offdiag element
  int count = 0;                                    //Variable for counting iterations

  R = zeros<mat>(n, n);                             //matrix of eigenvectors
  R.diag() += 1.0;

  while (fabs(max_offdiag) > eps && count < max_itr){
    max_offdiag = maxoffdiag(A,l,k,n);              //finds the max offdiagonal element
    rotation(A,R,k,l,n);                            //performs the jacobi rotation
    count++;
  }
  cout << "Iterations: " << count << "\n";



  uword idx = index_min(A.diag());
  GR = zeros<vec>(n);
  GR = R(span::all , idx);                        //Gets the groundvalue vector

  vec eigenvalues = sort(A.diag());               //Returns a vector of eigenvalues

  return eigenvalues;
}


vec eigvals_analytical(int n, double h) {
        // returns analytical eigenvalues of matrix A(
        double d = 2.0/(h*h);
        double a = -1.0/(h*h);
        vec lambda = zeros<vec>(n);
        for(int j = 1; j < n+1; j++)
                lambda[j-1] = d + 2.0*a*cos(j*M_PI/(n + 1.0));
        return lambda;
}
