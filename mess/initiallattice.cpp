

int p_bc(int i, int bound, int n_add){
  return (i+bound+n_add)%(bound);
}

void Energy_lattice(int n_Spin, mat &spin_mat, double &Enr, double &Mag)
{
  for(int i = 0; i < n_Spin; i++){
    for(int j = 0; j < n_Spin; j++){
      spin_mat(i,j) = 1.0;
      Mag += (double) spin_mat(i,j);
    }
  }

  for(int i = 0; i < n_Spin; i++){
    for(int j = 0; j < n_Spin; j++){
      Enr -= (double) spin_mat(i,j)*
      (spin_mat(p_bc(i, n_Spin, -1), j) +
      spin_mat(i ,p_bc(j, n_Spin, -1)));
    }
  }
}

map<double , double> Flip(double Tmp)
{
  map<double , double> accF;

  for (int de = -8; de <= 8; de+=4){
    accF.insert(pair<double , double>(de,exp(-de/Tmp)));
  }
  return accF;
}
