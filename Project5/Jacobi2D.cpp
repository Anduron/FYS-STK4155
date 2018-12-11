void JSolver(mat &A, mat &rhs, double dx, double dt, double tol, int n, double alpha) {
        double difference;
        mat A_temp = ones<mat>(n+2,n+2);

        double max_itr = pow(n,3);  // maximum iteration
        double itr = 0;                    // iteration counter

        for(int i = 1; i < n+2; i++) {
                A(i,0) = 0.0;
                A(i,n+1) = 0.0;
                A(0,i) = 0.0;
                A(n+1,i) = 0.0;
        }

        while((itr <= max_itr) && (difference > tol)) {
                A_temp = A;
                difference = 0;

                for(int i = 1; i < n+1; i++) {
                        for(int j = 1; j < n+1; j++) {
                                A(i,j) = dt*rhs(i,j) + A_temp(i,j) +
                                         alpha*(A_temp(i+1,j) + A_temp(i,j+1) -
                                                4*A_temp(i,j) + A_temp(i-1,j)+ A_temp(i,j-1) );

                                difference += fabs(A(i,j) - A_temp(i,j));
                        }
                }
                itr++;
        }
}
