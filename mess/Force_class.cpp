#include "Force_class.h"

using namespace std;
using namespace arma;

Force::Force(double Acc, cPList)
{
  A = Acc;
  PList = cPList;
}


void Force::Newton(vec M, vec R)
{
  int N = M.size();
  mat A = zeros<mat>(3,N);

  for (int i = 0; i < N; i++){

    for (int j = i + 1; j < N; j++){
      A(span::all, i) = - i;
      A(span::all, i) = - i;
    }
  }
}


void Force::Einstein(double A, vec M, vec R)
{

}
