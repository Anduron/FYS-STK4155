#include "Jacobi.hpp"

/* Jacobi iterative method for Backward Euler scheme. The solver takes the adresse
   of matrix u, which it changes over each iteration. The solver also takes rhs,
   the right hand side of the equation Au(t+dt)=u(t), which is unchanged during
   the iterations.
 */
void JSolver(double d, int n, double alpha, mat &u, mat rhs){
        double Qd;    // new potential, depends on debth
        mat u_old;    // u(t), used to calculate u(t+dt)

        int maxIter = 1000;   // maximum number of iterations
        int iter = 0;         // counter for iterations

        double diff = 1.0;
        double tol = 1E-10;

        double GYr = 3600*24*365*1E9;
        double lscale = 120000;
        double rho = 3.51E3;
        double Cp = 1000;
        double k = 2.5;
        double dt = 1/pow(118.0,2);
        double tempsc = 1300;
        double scale1 = (k*GYr)/(Cp*rho*lscale*lscale);  // specific heat capacity times the density
        double scale2 = (dt*GYr)/(Cp*rho*tempsc);
        //cout << scale1 << " " << scale2 << " ";

        while( iter < maxIter && diff > tol) {
                diff = 0;
                u_old = u;
                for(int i = 1; i < n+1; i++) {
                        if(i <= 20) {
                                Qd = 1.4E-6*scale2;
                        }
                        else if(i <= 40) {
                                Qd = 0.35E-6*scale2;
                        }
                        else if(i <= 120) {
                                Qd = 0.05E-6*scale2 + 0.50E-6*scale2;
                        }

                        //cout << Qd << endl;
                        for(int j = 1; j < n+1; j++) {
                                // Backward Euler
                                u(i,j) = (1.0/(1+4*alpha*scale1))*(alpha*scale1*(u_old(i+1,j) + u_old(i,j+1) +
                                                        u_old(i-1,j)+ u_old(i,j-1)) + Qd + rhs(i,j));
                                diff += abs(u(i,j) - u_old(i,j));
                        }
                }
                iter++;
        }
        cout << iter << endl;
}
