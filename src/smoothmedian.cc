// smoothmedian.cc
// c++ source code for creating smoothmedian.oct file using mkoctfile 
// in Octave 
//
// smoothmedian.oct is a function file for calculating a smooth 
// version of the median [1]
//
// M   = smoothmedian (x)
// ... = smoothmedian (x, dim)
// ... = smoothmedian (x, dim, Tol)      
// [M, SE] = smoothmedian (...)
//
// INPUT VARIABLES
// x (double) is the data vector or matrix
// dim (double) is the dimension (1 for columnwise, 2 for rowwise)
// Tol (double) sets the step size that will to stop optimization 
//
// OUTPUT VARIABLE
// M (double) is vector of the smoothed median
// SE (double) is a vector of standard errors of the smoothed median
//
// If x is a vector, find the univariate smoothed median (M) of x.
// If x is a matrix, compute the univariate smoothed median value
// for each column and return them in a row vector (default). The 
// argument dim defines which dimension to operate along. Arrays  
// of more than two dimensions are not currently supported. Tol 
// configures the stopping criteria, in terms of the absolute 
// change in the step size. By default, Tol = RANGE * 1e-4.
//
// The smoothed median is a slightly smoothed version of the ordinary 
// median and is an M-estimator that is both robust and efficient:
//
// | Asymptotic            | Mean |    Median  |    Median  |
// | properties            |      | (smoothed) | (ordinary) |
// |-----------------------|------|------------|------------|
// | Breakdown point       | 0.00 |      0.341 |      0.500 |
// | Pitman efficacy       | 1.00 |      0.865 |      0.637 |
//
// Smoothing the median is achieved by minimizing the following
// objective function:
//
//      S (M) = sum (((x(i) - M).^2 + (x(j) - M).^2).^ 0.5)
//             i < j
// 
// where i and j refers to the indices of the Cartesian product 
// of each column of x with itself. 
//
// This function minimizes the above objective function by finding 
// the root of the first derivative using a fast, but reliable, 
// Newton-Bisection hybrid algorithm. The tolerance (Tol) is the 
// maximum step size that is acceptable to break from optimization.
// Standard errors of the smoothed median can be estimated and
// returned by requesting the second output argument (SE).
//
// Bibliography:
// [1] Brown, Hall and Young (2001) The smoothed median and the
//      bootstrap. Biometrika 88(2):519-534
//
// Author: Andrew Charles Penn (2022)

#include <octave/oct.h>

DEFUN_DLD (smoothmedian, args, nargout, 
           " smoothmedian.oct is a function file for calculating a smoothed \n"\
           " version of the median [1] \n"\
           " \n"\
           " M   = smoothmedian (x) \n"\
           " ... = smoothmedian (x, dim) \n"\
           " ... = smoothmedian (x, dim, Tol) \n"\
           " [M, SE] = smoothmedian (...) \n"\
           " \n"\
           " INPUT VARIABLES \n"\
           " x (double) is the data vector or matrix \n"\
           " dim (double) is the dimension (1 for columnwise, 2 for rowwise) \n"\
           " Tol (double) sets the step size that will to stop optimization \n"\
           " \n"\
           " OUTPUT VARIABLE \n"\
           " M (double) is vector of the smoothed median \n"\
           " SE (double) is a vector of standard errors of the smoothed median \n"\
           " \n"\
           " If x is a vector, find the univariate smoothed median (M) of x. \n"\
           " If x is a matrix, compute the univariate smoothed median value \n"\
           " for each column and return them in a row vector. The argument dim \n"\
           " defines which dimension to operate along. Arrays of more than two \n"\
           " dimensions are not currently supported. Tol configures the stopping \n"\
           " criteria, in terms of the change in the step size. \n"\
           " \n"\
           " The smoothed median is a slightly smoothed version of the ordinary \n"\
           " median and is an M-estimator that is both robust and efficient: \n"\
           " \n"\
           " | Asymptotic            | Mean |    Median  |    Median  | \n"\
           " | properties            |      | (smoothed) | (ordinary) | \n"\
           " |-----------------------|------|------------|------------| \n"\
           " | Breakdown point       | 0.00 |      0.341 |      0.500 | \n"\
           " | Pitman efficacy       | 1.00 |      0.865 |      0.637 | \n"\
           " \n"\
           " Smoothing the median is achieved by minimizing the following \n"\
           " objective function: \n\n"\
           "      S (M) = sum (((x(i) - M).^2 + (x(j) - M).^2).^ 0.5) \n"\
           "             i < j \n\n"\
           " \n"\
           " where i and j refers to the indices of the Cartesian product \n"\
           " of each column of x with itself. \n"\
           " \n"\
           " This function minimizes the above objective function by finding \n"\
           " the root of the first derivative using a fast, but reliable, \n"\
           " Newton-Bisection hybrid algorithm. The tolerance (Tol) is the \n"\
           " maximum step size that is acceptable to break from optimization. \n"\
           " Standard errors of the smoothed median can be estimated and \n"\
           " returned by requesting the second output argument (SE). \n"\
           " \n"\
           " Bibliography: \n"\
           " [1] Brown, Hall and Young (2001) The smoothed median and the \n"\
           "      bootstrap. Biometrika 88(2):519-534 \n"\
           " \n"\
           " Author: Andrew Charles Penn (2022)")
 {

    // Input variable declaration
    Matrix x;
    short int dim;
    double Tol;
        
    // Input variables
    if (args.length () < 1) {
        print_usage ();
    } else {
        x = args(0).matrix_value();
    }
    if (args.length () < 2) {
        dim = 1;
    } else {
        dim = args(1).int_value();
    }
    
    // Check that there are no inf or nan values in the data
    if (x.any_element_is_inf_or_nan ()) {
        octave_stdout << "Error: x cannot contain Inf or NaN values\n";
        return octave_value ();
    }
    
    // If applicable, switch dimension
    if (dim > 1) {
        x = x.transpose ();
    }
      
    // Obtain data dimensions
    const dim_vector sz = x.dims ();
    short int m = sz(0);
    short int n = sz(1);
    short int l = m*(m-1);
    
    // Calculate basic statistics for each column of the data
    Matrix xmax = x.max ();
    Matrix xmin = x.min ();
    Matrix range = (xmax - xmin); // Range
    Matrix M = (xmax + xmin) / 2; // Mid-range
    
    // Create output variable for the standard errors
    dim_vector dv (1, n);
    Matrix SE (dv);

    // Create pointers so that we can more rapidly access elements of the matrices
    double *ptrX = x.fortran_vec ();
    double *ptrM = M.fortran_vec ();
    double *ptrXMIN = xmin.fortran_vec ();
    double *ptrXMAX = xmax.fortran_vec ();
    double *ptrRANGE = range.fortran_vec ();

    // Declare variables that we update in the loop with math assignment operators
    double T;
    double v;
    double U;
    
    // Loop through each column of the data 
    int MaxIter = 500;
    for (int k = 0; k < n ; k++) {
        
        if (args.length () < 3) {
            Tol = *(ptrRANGE + k) * 0.0001; 
        } else {
            Tol = args(2).double_value();
        }
        
        // Using the midrange as the starting value, find the smoothed median
        // Set initial bracket bounds
        double a = *(ptrXMIN + k); 
        double b = *(ptrXMAX + k);
        
        // Set initial value of free parameter to the midrange
        double p = *(ptrM + k);
               
        // Start iterations
        for (int Iter = 0; Iter < MaxIter ; Iter++) {
            T = 0;
            v = 0;
            U = 0;
            for (int j = 0; j < m ; j++) {
                for (int i = 0; i < j ; i++) {
                    double xi = *(ptrX + k * m + i);
                    double xj = *(ptrX + k * m + j);
                    // Calculate first derivative (T)
                    double D = pow (xi - p, 2) + pow (xj - p, 2);
                    double R = sqrt(D);
                    double t = (2 * p - xi - xj) / R;
                    T += t;
                    if (nargout > 1) {
                        // Compute v (required for standard error calculation later on)
                        v += pow (t / (m - 1), 2);
                    }
                    // Calculate second derivative (U)
                    U += pow (xi - xj, 2) * R / pow (D, 2);
                }
            }
            // Compute Newton step (fast quadratic convergence but unreliable)
            double step = T / U;
            // Evaluate convergence
            if (abs (step) < Tol) {
                break; // Break from optimization when converged to Tolerance 
            } else {
                // Update bracket bounds for Bisection
                if (T < -Tol) {
                    a = p;
                } else if (T > +Tol) {
                    b = p;
                }
                // Preview new value of the smoothed median
                double nwt = p - step;
                // Choose which method to use to update the smoothed median
                if (nwt > a && nwt < b) {
                    // Use Newton step if it is within bracket bounds
                    p = nwt;
                } else {
                    // Compute Bisection step (slow linear convergence but very safe)
                    p = 0.5 * (a + b);
                }
            }
            if (Iter == MaxIter) {
                octave_stdout << "Warning: Root finding failed to reach the specified tolerance.\n";
            }
        }
        
        // Assign parameter value that optimizes the objective function for the smoothed median
        M(k) = p;
        
        if (nargout > 1) {
            // Calculate standard error of the smoothed median
            double v0 = U / l;
            v /= m;
            SE(k) = sqrt (pow (v0,-2) * v);  
        }
    
    }
    
    // If applicable, switch dimension
    if (dim > 1) {
        M = M.transpose ();
        if (nargout > 1) {
            SE = SE.transpose (); 
        }
    }

    // Prepare output aruments to return to Octave
    octave_value_list retval;
    retval(0) = octave_value(M);
    retval(1) = octave_value(SE);
    
    return retval;
} 