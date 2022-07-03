// boot.cpp
// c++ source code for creating boot.mex file using mex as follows:
// 
// mex -compatibleArrayDims boot.cpp
//
// boot.mex is a function file for generating balanced bootstrap sample indices
//
// USAGE
// bootsam = boot (n, nboot)
// bootsam = boot (n, nboot, u)
// bootsam = boot (n, nboot, u, w)
//
// INPUT VARIABLES
// n (integer, int32) is the number of rows (of the data vector)
// nboot (integer, int32) is the number of bootstrap resamples
// u (boolean) for unbiased: false (for bootstrap) or true (for bootknife)
// w (double) is a weight vector of length n. 
//
// OUTPUT VARIABLE
// bootsam (integer, int32) is an n x nboot matrix of bootstrap resamples
//
// NOTES
// Uniform random numbers are generated by the Mersenne Twister 19937 generator.
// u is an optional input argument. The default is false. If u is true then 
// the sample index for omission in each bootknife resample is selected 
// systematically. If the remaining number of bootknife resamples is not 
// divisible by the sample size (n), then the sample index omitted is  
// selected randomly. 
// w is an optional input argument. The default is a vector of length n with
// each element equal to nboot (i.e. uniform weighting). Each element of w
// is the number of times that the corresponding index is represented in 
// bootsam. For example, if the second element is 500, then the value 2 will 
// will be assigned to 500 elements within bootsam. The sum of w should equal 
// n * nboot.
//
// Author: Andrew Charles Penn (2022)


#include "mex.h"
#include <random>
using namespace std;


void mexFunction (int nlhs, mxArray* plhs[],
                  int nrhs, const mxArray* prhs[]) 
{
  
    // Input variables
    if (nrhs < 2) {
        mexErrMsgTxt("function requires at least 2 scalar input arguments");
    }
    // First input argument
    const int n = *(mxGetPr(prhs[0]));
    if (mxGetNumberOfElements (prhs[0]) > 1) {
        mexErrMsgTxt("the first input argument must be scalar");
    }
    if (n <= 0) {
        mexErrMsgTxt("the first input argument must be a positive integer");
    }
    // Second input argument
    const int nboot = *(mxGetPr(prhs[1]));
    if (mxGetNumberOfElements (prhs[1]) > 1) {
        mexErrMsgTxt("the second input argument must be scalar");
    }
    if (nboot <= 0) {
        mexErrMsgTxt("the second input argument must be a positive integer");
    }
    // Third input argument
    bool u;
    if (nrhs < 3) {
        u = false;
    } else {
        u = *(mxGetPr(prhs[2]));
    }    
    
    // Output variables
    if (nlhs > 1) {
        mexErrMsgTxt("function can only return a single output arguments");
    }
    
    // Declare variables
    mwSize dims[2] = {n, nboot};
    plhs[0] = mxCreateNumericArray(2, dims, 
                mxINT32_CLASS, 
                mxREAL);             // Prepare array for bootstrap sample indices
    long long int N = n * nboot;     // Total counts of all sample indices
    long long int k;                 // Variable to store random number
    long long int d;                 // Counter for cumulative sum calculations
    long long int c[n];              // Counter for each of the sample indices 
    if (nrhs > 3) {
        // Assign user defined weights (counts)
        double *w = (double *) mxGetData (prhs[3]);
        if (mxGetNumberOfElements (prhs[3]) != n) {
            mexErrMsgTxt("weights must be a vector of length n");
        }
        long long int s = 0; 
        for (int i = 0; i < n ; i++)  {
            c[i] = w[i];     // Set each element in c to the specified weight
            s += c[i];
        }
        if (s != N) {
            mexErrMsgTxt("weights must add up to n * nboot");
        }
    } else {
        // Assign weights (counts) for uniform sampling
        for (int i = 0; i < n ; i++) {   
            c[i] = nboot;            // Set each element in c to nboot
        }
    }
    bool LOO = false;                // Leave-one-out (LOO) flag for the current bootstrap iteration (remains false if u is false)
    long long int m = 0;             // Counter for LOO sample index r (remains 0 if u is false) 
    int r = -1;                      // Sample index for LOO (remains -1 and is ignored if u is false)

    // Create pointer so that we can access elements of bootsam (i.e. plhs[0])
    int *ptr = (int *) mxGetData(plhs[0]);
    
    // Initialize random number generator
    random_device rd;
    seed_seq seed {rd(), rd(), rd(), rd()};
    mt19937 rng(seed);
    uniform_int_distribution<int> distr (0, n - 1);

    // Perform balanced sampling
    for (int b = 0; b < nboot ; b++) { 
        if (u) {    
            if ((b / n) == (nboot / n)) {
                r = distr (rng);      // random
            } else {
                r = b - (b / n) * n;  // systematic
            }
        }
        for (int i = 0; i < n ; i++) {
            if (u) {
                if (c[r] < N) {       // Only LOO if sample index r doesn't account for all remaining sampling counts
                    m = c[r];
                    c[r] = 0;
                    LOO = true;
                }
            }
            uniform_int_distribution<int> distk (0, N - m - 1);
            k = distk (rng); 
            d = c[0];
            for (int j = 0; j < n ; j++) { 
                if (k < d) {
                    ptr[b * n + i] = j + 1;
                    c[j] -= 1;
                    N -= 1;
                    break;
                } else {
                    d += c[j + 1];
                }
            }
            if (LOO == true) {
                c[r] = m;
                m = 0;
                LOO = false;
            }
        }   
    }    
    
    return;
    
}