#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "JacobiALG.h"

TEST_CASE("largest offdiag element"){

  int n = 7;
  double rho_0 = 0.0;
  double rho_n = 1.0;
  double h = (rho_n - rho_0)/double(n);

  mat testmat = toepliz(n, h);
  testmat.diag().fill(0);
  int k, l;

  mat testsize = abs(testmat);

  double expected = testsize.max();
  double maximum = maxoffdiag(testsize, k, l, n);
  REQUIRE(maximum == Approx(expected));
}

TEST_CASE("expected eigenvalues"){

  int n = 5;
  double rho_0 = 0.0;
  double rho_n = 1.0;
  double h = (rho_n - rho_0)/double(n);

  mat testmat = toepliz(n, h);
  testmat.diag().fill(0);

  mat eigenvectest;
  vec GR;

  vec eigenvalues = jacobi_method(testmat, eigenvectest, n, GR);

  vec Aeigval;
  mat Aeigvec;
  eig_sym(Aeigval, Aeigvec, testmat);

  for(int i = 0; i < n; i++) {
    REQUIRE(Aeigval[i]==Approx(eigenvalues[i]));
  }
}

//TEST_CASE(orthogonal_matrix){
//
//}
