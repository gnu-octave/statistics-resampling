% Advanced bootstrap confidence intervals: Performs bootstrap and calculates
% various types of confidence intervals from the resulting bootstrap statistics. 
%
% -- Function File: CI = bootci (NBOOT, BOOTFUN, D)
% -- Function File: CI = bootci (NBOOT, BOOTFUN, D1,...,DN)
% -- Function File: CI = bootci (NBOOT, {BOOTFUN, D}, NAME, VALUE)
% -- Function File: CI = bootci (NBOOT, {BOOTFUN, D1, ..., DN}, NAME, VALUE)
% -- Function File: CI = bootci (...,'type', TYPE)
% -- Function File: CI = bootci (...,'type', 'stud', 'nbootstd', NBOOTSTD)
% -- Function File: CI = bootci (...,'type', 'cal', 'nbootcal', NBOOTCAL)
% -- Function File: CI = bootci (...,'alpha', ALPHA)
% -- Function File: CI = bootci (...,'strata', STRATA)
% -- Function File: CI = bootci (...,'loo', LOO)
% -- Function File: CI = bootci (...,'seed', SEED)
% -- Function File: CI = bootci (...,'Options', PAROPT)
% -- Function File: [CI, BOOTSTAT] = bootci (...)
%
%     'CI = bootci (NBOOT, BOOTFUN, D)' draws NBOOT bootstrap resamples from
%     the rows of a data sample D and returns 95% confidence intervals (CI) for
%     the bootstrap statistics computed by BOOTFUN [1]. BOOTFUN is a function 
%     handle (e.g. specified with @), or a string indicating the function name. 
%     The third input argument, data D (a column vector or a matrix), is used
%     as input for BOOTFUN. The bootstrap resampling method yields first-order
%     balance [2-3].
%
%     'CI = bootci (NBOOT, BOOTFUN, D1,...,DN)' is as above except that the
%     third and subsequent numeric input arguments are data (column vectors
%     or matrices) that are used to create inputs for BOOTFUN.
%
%     'CI = bootci (NBOOT, {BOOTFUN, D}, NAME, VALUE)' is as above but includes
%     setting optional parameters using Name-Value pairs.
%
%     'CI = bootci (NBOOT, {BOOTFUN, D1, ..., DN}, NAME, VALUE)' is as above but
%     includes setting optional parameters using NAME-VALUE pairs.
%
%     bootci can take a number of optional parameters as NAME-VALUE pairs:
%
%     'CI = bootci (..., 'alpha', ALPHA)' where ALPHA sets the lower and upper 
%     bounds of the confidence interval(s). The value of ALPHA must be between
%     0 and 1. The nominal lower and upper percentiles of the confidence
%     intervals CI are then 100*(ALPHA/2)% and 100*(1-ALPHA/2)% respectively,
%     and nominal central coverage of the intervals is 100*(1-ALPHA)%. The
%     default value of ALPHA is 0.05.
%
%     'CI = bootci (..., 'type', TYPE)' computes bootstrap confidence interval 
%     CI using one of the following methods:
%      <> 'norm' or 'normal': Using bootstrap bias and standard error [4].
%      <> 'per' or 'percentile': Percentile method [1,4].
%      <> 'basic': Basic bootstrap method [1,4].
%      <> 'bca': Bias-corrected and accelerated method [5,6] (Default).
%      <> 'stud' or 'student': Studentized bootstrap (bootstrap-t) [1,4].
%      <> 'cal': Calibrated percentile method (by double bootstrap [7]).
%       Note that when BOOTFUN is the mean, the percentile, basic and bca
%       intervals are automatically expanded using Student's t-distribution in
%       order to improve coverage for small samples [8]. To compute confidence
%       the intervals for the mean without this correction for narrowness bias,
%       define the mean in BOOTFUN as an anonymous function (e.g. @(x) mean(x)).
%       The bootstrap-t method includes an additive correction to stabilize the
%       variance when the sample size is small [9].
%
%     'CI = bootci (..., 'type', 'stud', 'nbootstd', NBOOTSTD)' computes the
%     Studentized bootstrap confidence intervals CI, with the standard errors
%     of the bootstrap statistics estimated automatically using resampling
%     methods. NBOOTSTD is a positive integer value > 0 defining the number
%     of resamples. Standard errors are computed using NBOOTSTD bootstrap
%     resamples. The default value of NBOOTSTD is 100.
%
%     'CI = bootci (..., 'type', 'cal', 'nbootcal', NBOOTCAL)' computes the
%     calibrated percentile bootstrap confidence intervals CI, with the
%     calibrated percentiles of the bootstrap statistics estimated from NBOOTCAL
%     bootstrap data samples. NBOOTCAL is a positive integer value. The default
%     value of NBOOTCAL is 199.
%
%     'CI = bootci (..., 'strata', STRATA)' sets STRATA, which are identifiers
%     that define the grouping of the DATA rows for stratified bootstrap
%     resampling. STRATA should be a column vector or cell array with the same
%     number of rows as the DATA.
%
%     'CI = bootci (..., 'loo', LOO)' is a logical scalar that specifies whether
%     the resamples of size n should be obtained by sampling from the original
%     data (false) or from Leave-One-Out (LOO) jackknife samples (true) of the
%     data - otherwise known as bootknife resampling [10]. Default is false.
%
%     'CI = bootci (..., 'seed', SEED)' initialises the Mersenne Twister random
%     number generator using an integer SEED value so that bootci results are
%     reproducible.
%
%     'CI = bootci (..., 'Options', PAROPT)' specifies options that govern if
%     and how to perform bootstrap iterations using multiple processors (if the
%     Parallel Computing Toolbox or Octave Parallel package is available). This
%     argument is a structure with the following recognised fields:
%       <> 'UseParallel': If true, use parallel processes to accelerate
%                         bootstrap computations on multicore machines,
%                         specifically non-vectorized function evaluations,
%                         double bootstrap resampling and jackknife function
%                         evaluations. Default is false for serial computation.
%                         In MATLAB, the default is true if a parallel pool
%                         has already been started. 
%       <> 'nproc':       nproc sets the number of parallel processes (optional)
%
%     '[CI, BOOTSTAT] = bootci (...)' also returns the bootstrap statistics
%     used to calculate the confidence intervals CI.
%   
%     '[CI, BOOTSTAT, BOOTSAM] = bootci (...)' also returns BOOTSAM, a matrix 
%     of indices from the bootstrap. Each column in BOOTSAM corresponds to one 
%     bootstrap sample and contains the row indices of the values drawn from 
%     the nonscalar data argument to create that sample.
%
%  Bibliography:
%  [1] Efron, and Tibshirani (1993) An Introduction to the
%        Bootstrap. New York, NY: Chapman & Hall
%  [2] Davison et al. (1986) Efficient Bootstrap Simulation.
%        Biometrika, 73: 555-66
%  [3] Booth, Hall and Wood (1993) Balanced Importance Resampling
%        for the Bootstrap. The Annals of Statistics. 21(1):286-298
%  [4] Davison and Hinkley (1997) Bootstrap Methods and their Application.
%        (Cambridge University Press)
%  [5] Efron (1987) Better Bootstrap Confidence Intervals. JASA, 
%        82(397): 171-185 
%  [6] Efron, and Tibshirani (1993) An Introduction to the
%        Bootstrap. New York, NY: Chapman & Hall
%  [7] Hall, Lee and Young (2000) Importance of interpolation when
%        constructing double-bootstrap confidence intervals. Journal
%        of the Royal Statistical Society. Series B. 62(3): 479-491
%  [8] Hesterberg, Tim (2014), What Teachers Should Know about the 
%        Bootstrap: Resampling in the Undergraduate Statistics Curriculum, 
%        http://arxiv.org/abs/1411.5279
%  [9] Polansky (2000) Stabilizing bootstrap-t confidence intervals
%        for small samples. Can J Stat. 28(3):501-516
%  [10] Hesterberg T.C. (2004) Unbiasing the Bootstrap—Bootknife Sampling 
%        vs. Smoothing; Proceedings of the Section on Statistics & the 
%        Environment. Alexandria, VA: American Statistical Association.
%
%  bootci (version 2024.05.13)
%  Author: Andrew Charles Penn
%  https://www.researchgate.net/profile/Andrew_Penn/
%
%  Copyright 2019 Andrew Charles Penn
%  This program is free software: you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation, either version 3 of the License, or
%  (at your option) any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program.  If not, see http://www.gnu.org/licenses/


function [ci, bootstat, bootsam] = bootci (argin1, argin2, varargin)

  % Evaluate the number of function arguments
  if (nargin < 2)
    error (cat (2, 'bootci usage: ''bootci (NBOOT, {BOOTFUN, DATA},', ...
                   ' varargin)''; atleast 2 input arguments required'))
  end

  % Check if using MATLAB or Octave
  info = ver; 
  ISOCTAVE = any (ismember ({info.Name}, 'Octave'));

  % Apply defaults
  alpha = 0.05;
  type = 'bca';
  nbootstd = 100;
  nbootcal = 199;
  strata = [];
  loo = false;
  paropt = struct;
  paropt.UseParallel = false;
  if (~ ISOCTAVE)
    ncpus = feature ('numcores');
  else
    ncpus = nproc;
  end
  paropt.nproc = ncpus;

  % Assign input arguments to function variables
  nboot = argin1;
  argin3 = varargin;
  narg = numel (argin3);
  if (narg > 1)
    name = argin3{end - 1};
    value = argin3{end};
    while (ischar (name))
      switch (lower (name))
        case {'options', 'option'}
          paropt = value;
        case 'alpha'
          alpha = value;
        case 'type'
          type = value;
        case 'nbootstd'
          nbootstd = value;
        case 'nbootcal'
          nbootcal = value;
        case 'strata'
          strata = value;
        case 'loo'
          loo = value;
        case'seed'
          seed = value;
          % Initialise the random number generator with the seed
          boot (1, 1, true, seed);
        otherwise
          error ('bootci: Unrecognised input argument to bootci')
      end
      argin3 = argin3(1:end-2);
      narg = numel (argin3);
      if (narg < 2)
        break
      end
      name = argin3{end - 1};
      value = argin3{end};
    end
  end
  if (iscell (argin2))
    bootfun = argin2{1};
    if (numel (argin2) > 2)
      data = argin2(2:end);
      n = size (data{1}, 1);
    else
      data = argin2{2};
      n = size (data, 1);
    end
  else
    bootfun = argin2;
    if (numel (argin3) > 1)
      data = argin3;
      n = size (data{1}, 1);
    else
      data = argin3{1};
      n = size (data, 1);
    end
  end
  if (paropt.UseParallel)
    if (isfield (paropt, 'nproc'))
      ncpus = paropt.nproc;
    else
      if (ISOCTAVE)
        ncpus = inf;
      else
        ncpus = [];
      end
    end
  else
    ncpus = 0;
  end

  % Error checking
  if (numel(alpha) > 1)
    error ('bootci: ALPHA must be a scalar value');
  end
  if (~ isa (alpha,'numeric'))
    error ('bootci: ALPHA must be a numeric');
  end
  if (any ((alpha < 0) | (alpha > 1)))
    error ('bootci: ALPHA must be a value between 0 and 1');
  end
  if (~ isa (nboot, 'numeric'))
    error ('bootci: NBOOT must be numeric');
  end
  if (numel (nboot) > 1)
    error ('bootci: NBOOT must be a positive integer');
  end
  if (nboot ~= abs (fix (nboot)))
    error ('bootci: NBOOT must contain positive integers');
  end
  if (~ isa (nbootstd, 'numeric'))
    error ('bootci: NBOOTSTD must be numeric');
  end
  if (numel (nbootstd) > 1)
    error ('bootci: NBOOTSTD must be a scalar value');
  end
  if (nbootstd < 1)
    error ('bootci: NBOOTSTD must be an integer > 0');
  end  
  if (~ isa (nbootcal, 'numeric'))
    error ('bootci: NBOOTCAL must be numeric');
  end
  if (numel (nbootcal) > 1)
    error ('bootci: NBOOTCAL must be a scalar value');
  end
  if (nbootcal ~= abs (fix (nbootcal)))
    error ('bootci: NBOOTCAL must be a positive integer');
  end
  % If applicable, check we have parallel computing capabilities
  if (ncpus > 1)
    if (ISOCTAVE)
      software = pkg ('list');
      names = cellfun (@(S) S.name, software, 'UniformOutput', false);
      status = cellfun (@(S) S.loaded, software, 'UniformOutput', false);
      index = find (~ cellfun (@isempty, regexpi (names, '^parallel')));
      if (~ isempty (index))
        if (~ logical (status{index}))
          ncpus = 0;
        end
      else
        ncpus = 0;
      end
      if (ncpus == 0)
        % OCTAVE Parallel Computing Package is not installed or loaded
        warning ('bootci:parallel', ...
                 cat (2, 'Parallel package is not installed and/or loaded.', ...
                         ' Falling back to serial processing.'))
      end
    else
      info = ver; 
      if (~ ismember ('Parallel Computing Toolbox', {info.Name}))
        ncpus = 0;
      end
      if (ncpus == 0)
        % MATLAB Parallel Computing Toolbox is not installed or loaded
        warning ('bootci:parallel', ...
                 cat (2, 'Parallel Computing Toolbox is installed and/or', ...
                         ' loaded. Falling back to serial processing.'))
      end
    end
  end
  if ( (~ isscalar (loo)) || (~ islogical (loo)) )
    error ('bootci: LOO must be a logical scalar value')
  end

  % Apply interval type
  switch (lower (type))
    case {'per', 'perc', 'percentile', 'basic', 'norm', 'normal'}
      % Do nothing
    case {'bca', 'stud', 'student'}
      % Set quantiles directly to calculate percentile intervals
      alpha = [alpha / 2, 1 - alpha / 2];
    case 'cal'
      % Set quantiles directly to calibrate confidence intervals
      alpha = [alpha / 2, 1 - alpha / 2];
      nboot = cat (2, nboot, nbootcal);
    otherwise
      error ('bootci: Interval TYPE not supported')
  end

  % Parse input arguments to bootknife function and compute confidence intervals
  ci = [];
  switch (lower (type))
 
    case {'norm', 'normal'}
      % Intervals constructed based on bootstrap estimates of bias and standard
      % error assumming that the sampling distribution is Normal
      [stats, bootstat, bootsam] = bootknife (data, nboot, bootfun, NaN, ...
                                   strata, ncpus, [], [], ISOCTAVE, true, loo);
      stdnorminv = @(p) sqrt (2) * erfinv (2 * p-1);
      z = stdnorminv (alpha / 2);
      ci = cat (2, stats.original - stats.bias + stats.std_error * z, ...
                   stats.original - stats.bias - stats.std_error * z).';

    case 'basic'
      % The basic bootstrap method.
      % Center bootstrap statistics
      [stats, bootstat, bootsam] = bootknife (data, nboot, bootfun, alpha, ...
                                   strata, ncpus, [], [], ISOCTAVE, true, loo);
      ci = cat (2, 2 * stats.original - stats.CI_upper, ...
                   2 * stats.original - stats.CI_lower).';

    case {'stud', 'student'}
      % Use bootstrap-t method with variance stabilization for small samples
      % Polansky (2000) Can J Stat. 28(3):501-516
      [stats, bootstat, bootsam] = bootknife (data, nboot, bootfun, NaN, ...
                                   strata, ncpus, [], [], ISOCTAVE, true, loo);
      % Automatically estimate standard errors of the bootstrap statistics
      % Using bootstrap resampling
      if (iscell (data))
        % If DATA is a cell array of equal size colunmn vectors, convert the
        % cell array to a matrix and define function to calculate an estimate
        % of the standard error using bootstrap resampling
        szx = cellfun (@(x) size (x, 2), data);
        data = [data{:}];
        cellfunc = @(bootsam) bootknife ( ...
                          mat2cell (data (bootsam,:), n, szx), nbootstd, ...
                          bootfun, NaN, strata, 0, [], [], ISOCTAVE, true, loo);
      else
        cellfunc = @(bootsam) bootknife (data (bootsam,:), nbootstd, ...
                          bootfun, NaN, strata, 0, [], [], ISOCTAVE, true, loo);
      end
      if (ncpus > 1)
        if (ISOCTAVE)
          % Octave
          % Set unique random seed for each parallel thread
          pararrayfun (ncpus, @boot, 1, 1, false, 1:ncpus);
          bootout = parcellfun (ncpus, cellfunc, num2cell (bootsam, 1), ...
                                'UniformOutput', false);
        else
          % MATLAB
          % Set unique random seed for each parallel thread
          parfor i = 1:ncpus; boot (1, 1, false, i); end
          % Perform inner layer of resampling
          bootout = cell (1, nboot(1));
          parfor b = 1:nboot(1); bootout{b} = cellfunc (bootsam(:,b)); end
        end
      else
        bootout = cellfun (cellfunc, num2cell (bootsam, 1), ...
                           'UniformOutput', false);
      end
      se = cell2mat (cellfun (@(S) S.std_error, bootout, ...
                              'UniformOutput', false));
      % Compute additive constant to stabilize the variance
      a = n^(-3/2) * stats.std_error;
      % Calculate Studentized bootstrap statistics
      T = bsxfun (@minus, bootstat, stats.original) ./ bsxfun (@plus, se, a);
      % Calculate intervals from empirical distribution of the Studentized
      % bootstrap statistics
      m = size (bootstat, 1);
      ci = zeros (m, 2);
      for j = 1:m
        [t, cdf] = bootcdf (T(j,:), true, 1);
        ci(j,1) = stats.original(j) - stats.std_error(j) * ...
                                interp1 (cdf, t, alpha(2), 'linear', max (t));
        ci(j,2) = stats.original(j) - stats.std_error(j) * ...
                                interp1 (cdf, t, alpha(1), 'linear', min (t));
      end
      ci = ci.';

    otherwise

      % Other interval types are natively supported in bootknife function
      % Use undocumented input argument 'loo' = false for simple bootstrap
      % resampling (instead of bootknife resampling)
      [stats, bootstat, bootsam] = bootknife (data, nboot, bootfun, alpha, ...
                                    strata, ncpus, [], [], ISOCTAVE, true, loo);

  end

  % Format output to be consistent with MATLAB's bootci
  if (isempty (ci))
    ci = cat (2, stats.CI_lower, stats.CI_upper).';
  end
  bootstat = bootstat.';

end

%--------------------------------------------------------------------------

%!demo
%!
%! % Input univariate dataset
%! data = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%!         0 33 28 34 4 32 24 47 41 24 26 30 41].';
%!
%! % 95% BCa bootstrap confidence intervals for the mean
%! ci = bootci (1999, @mean, data)
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input univariate dataset
%! data = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%!         0 33 28 34 4 32 24 47 41 24 26 30 41].';
%!
%! % 95% calibrated percentile bootstrap confidence intervals for the mean
%! ci = bootci (1999, {@mean, data}, 'type', 'cal','nbootcal',199)
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input univariate dataset
%! data = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%!         0 33 28 34 4 32 24 47 41 24 26 30 41].';
%!
%! % 95% calibrated percentile bootstrap confidence intervals for the median
%! % with smoothing
%! ci = bootci (1999, {@smoothmedian, data}, 'type', 'cal', 'nbootcal', 199)
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input univariate dataset
%! data = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%!         0 33 28 34 4 32 24 47 41 24 26 30 41].';
%!
%! % 90% percentile bootstrap confidence intervals for the variance
%! ci = bootci (1999, {{@var,1}, data}, 'type', 'per', 'alpha', 0.1)
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input univariate dataset
%! data = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%!         0 33 28 34 4 32 24 47 41 24 26 30 41].';
%!
%! % 90% BCa bootstrap confidence intervals for the variance
%! ci = bootci (1999, {{@var,1}, data}, 'type', 'bca', 'alpha', 0.1)
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input univariate dataset
%! data = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%!         0 33 28 34 4 32 24 47 41 24 26 30 41]';
%!
%! % 90% Studentized bootstrap confidence intervals for the variance
%! ci = bootci (1999, {{@var,1}, data}, 'type', 'stud', ...
%!                                              'nbootstd', 50, 'alpha', 0.1)
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input univariate dataset
%! data = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%!         0 33 28 34 4 32 24 47 41 24 26 30 41].';
%!
%! % 90% calibrated percentile bootstrap confidence intervals for the variance
%! ci = bootci (1999, {{@var,1}, data}, 'type', 'cal', 'nbootcal', ...
%!              199, 'alpha', 0.1)
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input bivariate dataset
%! x = [2.12,4.35,3.39,2.51,4.04,5.1,3.77,3.35,4.1,3.35, ...
%!      4.15,3.56, 3.39,1.88,2.56,2.96,2.49,3.03,2.66,3].';
%! y  = [2.47,4.61,5.26,3.02,6.36,5.93,3.93,4.09,4.88,3.81, ...
%!       4.74,3.29,5.55,2.82,4.23,3.23,2.56,4.31,4.37,2.4].';
%!
%! % 95% BCa bootstrap confidence intervals for the correlation coefficient
%! ci = bootci (1999, @cor, x, y)
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%! 
%! % Calculating confidence intervals for the coefficients from logistic 
%! % regression using an example with an ordinal response from:
%! % https://uk.mathworks.com/help/stats/mnrfit.html
%! 
%! %>>>>>>>>> This code block must be run first in Octave only >>>>>>>>>>>>
%!
%! try
%!   pkg load statistics
%!   load carbig
%!   info = ver;
%!   if ( str2num ({info.Version}{strcmp({info.Name},'statistics')}(1:3)) < 1.5)
%!     error ('statistics package version must be > 1.5')
%!   end
%!   if (~ exist ('mnrfit', 'file'))
%!     % Octave Statistics package does not currently have the mnrfit function,
%!     % so we will use it's logistic_regression function for fitting ordinal
%!     % models instead. 
%!     function [B, DEV] = mnrfit (X, Y, varargin)
%!       % Note that if the outcome has more than two levels, the
%!       % logistic_regression function is only suitable when the
%!       % outcome is ordinal, so we would need to use append 'model',
%!       % 'ordinal' as a name-value pair in MATLAB when executing
%!       % it's mnrfit function (see below)
%!       [INTERCEPT, SLOPE, DEV] = logistic_regression (Y - 1, X, false);
%!       B = cat (1, INTERCEPT, SLOPE);
%!     end
%!   end
%!   stats_pkg = true;
%! catch
%!   stats_pkg = false;
%!   fprintf ('\nSkipping this demo...')
%!   fprintf ('\nRequired features of the statistics package not found.\n\n');
%! end
%!
%! %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%!
%! if (stats_pkg)
%!
%!   %>>>>>>>>>>>>>>>>>>> This code block is the demo >>>>>>>>>>>>>>>>>>>>>>
%!
%!   % This demo requires the statistics package in Octave (equivalent to
%!   % the Statistics and Machine Learning Toolbox in Matlab)
%!
%!   % Create the dataset
%!   load carbig
%!   X = [Acceleration Displacement Horsepower Weight];
%!
%!   % The responses 1 - 4 correspond to the following classification:
%!   % 1:  9 - 19 miles per gallon
%!   % 2: 19 - 29 miles per gallon
%!   % 3: 29 - 39 miles per gallon
%!   % 4: 39 - 49 miles per gallon
%!   miles = [1,1,1,1,1,1,1,1,1,1,NaN,NaN,NaN,NaN,NaN,1,1,NaN,1,1,2,2,1,2, ...
%!            2,2,2,2,2,2,2,1,1,1,1,2,2,2,2,NaN,2,1,1,2,1,1,1,1,1,1,1,1,1, ...
%!            2,2,1,2,2,3,3,3,3,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,2,1,1,1,1, ...
%!            1,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,1,1,1, ...
%!            1,1,2,2,2,1,2,2,2,1,1,3,2,2,2,1,2,2,1,2,2,2,1,3,2,3,2,1,1,1, ...
%!            1,1,1,1,1,3,2,2,3,3,2,2,2,2,2,3,2,1,1,1,1,1,1,1,1,1,1,1,2,2, ...
%!            1,3,2,2,2,2,2,2,1,3,2,2,2,2,2,3,2,2,2,2,2,1,1,1,1,2,2,2,2,3, ...
%!            2,3,3,2,1,1,1,3,3,2,2,2,1,2,2,1,1,1,1,1,3,3,3,2,3,1,1,1,1,1, ...
%!            2,2,1,1,1,1,1,3,2,2,2,3,3,3,3,2,2,2,4,3,3,4,3,2,2,2,2,2,2,2, ...
%!            2,2,2,2,1,1,2,1,1,1,3,2,2,3,2,2,2,2,2,1,2,1,3,3,2,2,2,2,2,1, ...
%!            1,1,1,1,1,2,1,3,3,3,2,2,2,2,2,3,3,3,3,2,2,2,3,4,3,3,3,2,2,2, ...
%!            2,3,3,3,3,3,4,2,4,4,4,3,3,4,4,3,3,3,2,3,2,3,2,2,2,2,3,4,4,3, ...
%!            3,3,3,3,3,3,3,3,3,3,3,3,3,2,NaN,3,2,2,2,2,2,1,2,2,3,3,3,2,2, ...
%!            2,3,3,3,3,3,3,3,3,3,3,3,2,3,2,2,3,3,2,2,4,3,2,3]';
%!
%!   % Model coefficients from logistic regression
%!   B = mnrfit (X, miles, 'model', 'ordinal');
%!
%!   % Bootsrap confidence intervals for each logistic regression coefficient
%!   ci = bootci (1999, @(X, miles) mnrfit (X, miles, 'model', 'ordinal'), ...
%!                X, miles);
%!   [B, ci']
%!
%!   % Where the first 3 rows are the intercept terms, and the last 4 rows
%!   % are the slope coefficients. For each predictor, the slope coefficient
%!   % corresponds to how a unit change in the predictor impacts on the odds,
%!   % which are proportional across the (ordered) catagories, where each
%!   % log-odds in each case is:
%!   %
%!   %       ln ( ( P[below] ) / ( P[above] ) )
%!   %
%!   % i.e. in mnrfit, the reference class is the higher of the two classes.
%!   % Therefore, a positive slope value indicates that a unit increase in the
%!   % predictor increases the odds of running at fewer miles per gallon.
%!
%!   % Note that ordinal and multinomial logistic regression (appropriate
%!   % for ordinal and nominal responses respectively) would be equivalent
%!   % for any binary outcome
%!
%!   %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%! 
%! end
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input dataset
%! y = randn (20,1); x = randn (20,1); X = [ones(20,1), x];
%!
%! % 95% BCa confidence interval for regression coefficients 
%! ci = bootci (1999, @mldivide, X, y)
%!
%! % N.B: Consider using either the 'bootwild', 'bootbayes' or 'bootlm'
%! % functions for bootstrapping linear regression problems instead
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Spatial Test Data from Table 14.1 of Efron and Tibshirani (1993)
%! % An Introduction to the Bootstrap in Monographs on Statistics and Applied 
%! % Probability 57 (Springer)
%!
%! % AIM:
%! % To construct 90% nonparametric bootstrap confidence intervals for var(A,1)
%! % var(A,1) = 171.5
%! % Exact intervals based on Normal theory are [118.4, 305.2].
%!
%! % Calculations using Matlab's 'Statistics and Machine Learning toolbox'
%! % (R2020b)
%! %
%! % A = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%! %      0 33 28 34 4 32 24 47 41 24 26 30 41].';
%! % varfun = @(A) var(A, 1);
%! % rng('default'); % For reproducibility
%! % rng('default'); ci1 = bootci (19999,{varfun,A},'alpha',0.1,'type','norm');
%! % rng('default'); ci2 = bootci (19999,{varfun,A},'alpha',0.1,'type','per');
%! % rng('default'); ci4 = bootci (19999,{varfun,A},'alpha',0.1,'type','bca');
%! % rng('default'); ci5 = bootci (19999,{varfun,A},'alpha',0.1,'type','stud');
%! %
%! % Summary of results from Matlab's 'Statistics and Machine Learning toolbox'
%! % (R2020b)
%! %
%! % method             |   0.05 |   0.95 | length | shape |  
%! % -------------------|--------|--------|--------|-------|
%! % ci1 - normal       |  108.9 |  247.4 |  138.5 |  1.21 |
%! % ci2 - percentile   |   97.6 |  235.8 |  138.2 |  0.87 |
%! % ci4 - BCa          |  114.9 |  260.5 |  145.6 |  1.57 |*
%! % ci5 - bootstrap-t  |   46.7 |  232.5 |  185.8 |  0.49 |** 
%! % -------------------|--------|--------|--------|-------|
%! % parametric - exact |  118.4 |  305.2 |  186.8 |  2.52 |
%! %
%! % * Bug in the fx0 subfunction of MathWorks MATLAB bootci function
%! % ** Bug in the bootstud subfunction of MathWorks MATLAB bootci
%!
%! % Calculations using the 'boot' and 'bootstrap' packages in R
%! % 
%! % library (boot)       # Functions from Davison and Hinkley (1997)
%! % A <- c(48,36,20,29,42,42,20,42,22,41,45,14,6,
%! %         0,33,28,34,4,32,24,47,41,24,26,30,41);
%! % n <- length(A)
%! %  var.fun <- function (d, i) { 
%! %        # Function to compute the population variance
%! %        n <- length (d); 
%! %        return (var (d[i]) * (n - 1) / n) };
%! %  boot.fun <- function (d, i) {
%! %        # Compute the estimate
%! %        t <- var.fun (d, i);
%! %        # Compute sampling variance of the estimate using Tukey's jackknife
%! %        n <- length (d);
%! %        U <- empinf (data=d[i], statistic=var.fun, type="jack", stype="i");
%! %        var.t <- sum (U^2 / (n * (n - 1)));
%! %        return ( c(t, var.t) ) };
%! % set.seed(1)
%! % var.boot <- boot (data=A, statistic=boot.fun, R=19999, sim='balanced')
%! % ci1 <- boot.ci (var.boot, conf=0.90, type="norm")
%! % ci2 <- boot.ci (var.boot, conf=0.90, type="perc")
%! % ci3 <- boot.ci (var.boot, conf=0.90, type="basic")
%! % ci4 <- boot.ci (var.boot, conf=0.90, type="bca")
%! % ci5 <- boot.ci (var.boot, conf=0.90, type="stud")
%! %
%! % library (bootstrap)  # Functions from Efron and Tibshirani (1993)
%! % set.seed(1); 
%! % ci4a <- bcanon (A, 19999, var.fun, alpha=c(0.05,0.95))
%! % set.seed(1); 
%! % ci5a <- boott (A, var.fun, nboott=19999, nbootsd=499, perc=c(.05,.95))
%! %
%! % Summary of results from 'boot' and 'bootstrap' packages in R
%! %
%! % method             |   0.05 |   0.95 | length | shape |  
%! % -------------------|--------|--------|--------|-------|
%! % ci1  - normal      |  109.6 |  246.7 |  137.1 |  1.22 |
%! % ci2  - percentile  |   97.9 |  234.8 |  136.9 |  0.86 |
%! % ci3  - basic       |  108.3 |  245.1 |  136.8 |  1.16 |
%! % ci4  - BCa         |  116.0 |  260.7 |  144.7 |  1.60 |
%! % ci4a - BCa         |  115.8 |  260.6 |  144.8 |  1.60 |
%! % ci5  - bootstrap-t |  112.0 |  291.8 |  179.8 |  2.02 |
%! % ci5a - bootstrap-t |  116.1 |  290.9 |  174.8 |  2.16 |
%! % -------------------|--------|--------|--------|-------|
%! % parametric - exact |  118.4 |  305.2 |  186.8 |  2.52 |
%! 
%! % Calculations using the 'statistics-resampling' package for Octave/Matlab
%! %
%! % A = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%! %      0 33 28 34 4 32 24 47 41 24 26 30 41].';
%! % ci1 = bootci (19999,{{@var,1},A},'alpha',0.1,'type','norm','seed',1);
%! % ci2 = bootci (19999,{{@var,1},A},'alpha',0.1,'type','per','seed',1);
%! % ci3 = bootci (19999,{{@var,1},A},'alpha',0.1,'type','basic','seed',1);
%! % ci4 = bootci (19999,{{@var,1},A},'alpha',0.1,'type','bca','seed',1);
%! % ci5 = bootci (19999,{{@var,1},A},'alpha',0.1,'type','stud',...
%! %                                              'nbootstd',100,'seed',1);
%! % ci6 = bootci (19999,{{@var,1},A},'alpha',0.1,'type','cal', ...
%! %                                              'nbootcal',499,'seed',1);
%! %
%! % Summary of results from 'statistics-resampling' package for Octave/Matlab
%! %
%! % method             |   0.05 |   0.95 | length | shape |  
%! % -------------------|--------|--------|--------|-------|
%! % ci1 - normal       |  108.9 |  247.4 |  138.5 |  1.22 |
%! % ci2 - percentile   |   98.2 |  235.1 |  136.9 |  0.87 |
%! % ci3 - basic        |  108.0 |  244.8 |  136.9 |  1.15 |
%! % ci4 - BCa          |  116.2 |  259.6 |  143.4 |  1.60 |
%! % ci5 - bootstrap-t  |  114.0 |  289.5 |  175.5 |  2.05 |
%! % ci6 - calibrated   |  115.3 |  276.9 |  161.6 |  1.87 |
%! % -------------------|--------|--------|--------|-------|
%! % parametric - exact |  118.4 |  305.2 |  186.8 |  2.52 |
%! %
%! % Simulation results for constructing 90% confidence intervals for the
%! % variance of a population N(0,1) from 1000 random samples of size 26
%! % (analagous to the situation above). Simulation performed using the
%! % bootsim script with nboot of 1999.
%! %
%! % method               | coverage |  lower |  upper | length | shape |
%! % ---------------------|----------|--------|--------|--------|-------|
%! % normal               |    81.5% |   3.0% |  15.5% |   0.77 |  1.21 |
%! % percentile           |    81.5% |   0.9% |  17.6% |   0.76 |  0.91 |
%! % basic                |    81.1% |   2.5% |  16.4% |   0.78 |  1.09 |
%! % BCa                  |    83.9% |   5.2% |  10.9% |   0.83 |  1.84 |
%! % bootstrap-t          |    89.2% |   4.3% |   6.5% |   0.99 |  2.15 |
%! % calibrated           |    88.9% |   3.7% |   7.4% |   0.89 |  2.04 |
%! % ---------------------|----------|--------|--------|--------|-------|
%! % parametric - exact   |    90.8% |   3.7% |   5.5% |   0.99 |  2.52 |

%!test
%! % Test for errors when using some different functionalities of bootci
%! warning ('off', 'bootknife:parallel')
%! try
%!   y = randn (20, 1); 
%!   bootci (1999, 'mean', y);
%!   bootci (1999, @mean, y);
%!   bootci (1999, @mean, y, 'alpha', 0.1);
%!   bootci (1999, {'mean', y}, 'alpha', 0.1);
%!   bootci (1999, {'mean', y}, 'strata', []);
%!   bootci (1999, {'mean', y}, 'loo', false);
%!   bootci (1999, {'mean', y}, 'loo', true);
%!   bootci (1999, {@mean, y}, 'alpha', 0.1);
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'seed', 1);
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'type', 'norm');
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'type', 'per');
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'type', 'basic');
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'type', 'bca');
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'type', 'stud');
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'type', 'stud', 'nbootstd', 100);
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'type', 'cal');
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'type', 'cal', 'nbootcal', 199);
%!   g = reshape (repmat ([1:5], 4, 1), 20, []);
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'strata', []);
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'type', 'norm', 'strata', g);
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'type', 'per', 'strata', g);
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'type', 'basic', 'strata', g);
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'type', 'bca', 'strata', g);
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'type', 'stud', 'strata', g);
%!   bootci (1999, {@mean, y}, 'alpha', 0.1, 'type', 'cal', 'strata', g);
%!   Y = randn (20); 
%!   bootci (1999, 'mean', Y);
%!   bootci (1999, @mean, Y);
%!   bootci (1999, @mean, Y, 'alpha', 0.1);
%!   bootci (1999, {'mean', Y}, 'alpha', 0.1);
%!   bootci (1999, {@mean, Y}, 'alpha', 0.1);
%!   bootci (1999, {@mean, Y}, 'alpha', 0.1, 'strata', g);
%!   bootci (1999, {@mean, Y}, 'alpha', 0.1, 'seed', 1);
%!   bootci (1999, {@mean, Y}, 'alpha', 0.1, 'type', 'norm');
%!   bootci (1999, {@mean, Y}, 'alpha', 0.1, 'type', 'per');
%!   bootci (1999, {@mean, Y}, 'alpha', 0.1, 'type', 'basic');
%!   bootci (1999, {@mean, Y}, 'alpha', 0.1, 'type', 'bca');
%!   bootci (1999, {@mean, Y}, 'alpha', 0.1, 'type', 'stud');
%!   bootci (1999, {@mean, Y}, 'alpha', 0.1, 'type', 'cal');
%!   y = randn (20,1); x = randn (20,1); X = [ones(20,1),x];
%!   bootci (1999, @cor, x, y);
%!   bootci (1999, {@cor, x, y}, 'strata', g);
%!   bootci (1999, @mldivide, X, y);
%!   bootci (1999, @mldivide, X, y, 'alpha', 0.1);
%!   bootci (1999, {@mldivide, X, y}, 'alpha', 0.1);
%!   bootci (1999, {@mldivide, X, y}, 'alpha', 0.1, 'type', 'norm');
%!   bootci (1999, {@mldivide, X, y}, 'alpha', 0.1, 'type', 'per');
%!   bootci (1999, {@mldivide, X, y}, 'alpha', 0.1, 'type', 'basic');
%!   bootci (1999, {@mldivide, X, y}, 'alpha', 0.1, 'type', 'bca');
%!   bootci (1999, {@mldivide, X, y}, 'alpha', 0.1, 'type', 'stud');
%!   bootci (1999, {@mldivide, X, y}, 'alpha', 0.1, 'type', 'cal');
%! catch
%!   warning ('on', 'bootknife:parallel')
%!   rethrow (lasterror)
%! end
%! warning ('on', 'bootknife:parallel')

%!test
%! % Spatial Test Data from Table 14.1 of Efron and Tibshirani (1993)
%! % An Introduction to the Bootstrap in Monographs on Statistics and Applied 
%! % Probability 57 (Springer)
%! A = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%!      0 33 28 34 4 32 24 47 41 24 26 30 41].';
%!
%! % Nonparametric 90% percentile confidence intervals (single bootstrap)
%! % Table 14.2 percentile intervals are 100.8 - 233.9
%! ci = bootci(1999,{{@var,1},A},'alpha',0.1,'type','per','seed',1);
%! if (isempty (regexp (which ('boot'), 'mex$')))
%!   % test boot m-file result
%!   assert (ci(1), 97.67455621301775, 1e-07);
%!   assert (ci(2), 233.1553254437869, 1e-07);
%! end
%!
%! % Nonparametric 90% BCa confidence intervals (single bootstrap)
%! % Table 14.2 BCa intervals are 115.8 - 259.6
%! ci = bootci(1999,{{@var,1},A},'alpha',0.1,'type','bca','seed',1);
%! if (isempty (regexp (which ('boot'), 'mex$')))
%!   % test boot m-file result
%!   assert (ci(1), 115.9276292059273, 1e-07);
%!   assert (ci(2), 255.789475809047, 1e-07);
%! end
%!
%! % Nonparametric 90% bootstrap-t confidence intervals (double bootstrap)
%! ci = bootci(1999,{{@var,1},A},'alpha',0.1,'type','stud', ...
%!                                                  'nbootstd',100,'seed',1);
%! if (isempty (regexp (which ('boot'), 'mex$')))
%!   % test boot m-file result
%!   assert (ci(1), 114.4602660067695, 1e-07);
%!   assert (ci(2), 288.7667036808156, 1e-07);
%! end
%!
%! % Nonparametric 90% calibrated percentile confidence intervals
%! % (double bootstrap)
%! ci = bootci(1999,{{@var,1},A},'alpha',0.1,'type','cal',...
%!                                                  'nbootcal',199,'seed',1);
%! if (isempty (regexp (which ('boot'), 'mex$')))
%!   % test boot m-file result
%!   assert (ci(1), 116.1158249047411, 1e-07);
%!   assert (ci(2), 277.5572125379819, 1e-07);
%! end
%!
%! % Exact intervals based on normal theory are 118.4 - 305.2 (Table 14.2)

%!test
%! % Law school data from Table 3.1 of Efron and Tibshirani (1993)
%! % An Introduction to the Bootstrap in Monographs on Statistics and Applied 
%! % Probability 57 (Springer)
%! LSAT = [576 635 558 578 666 580 555 661 651 605 653 575 545 572 594].';
%! GPA = [3.39 3.3 2.81 3.03 3.44 3.07 3 3.43 ...
%!        3.36 3.13 3.12 2.74 2.76 2.88 2.96].';
%!
%! % Nonparametric 90% percentile confidence intervals (single bootstrap)
%! % Percentile intervals on page 266 are 0.524 - 0.928
%! ci = bootci(1999,{@cor,LSAT,GPA},'alpha',0.1,'type','per','seed',1);
%! if (isempty (regexp (which ('boot'), 'mex$')))
%!   % test boot m-file result
%!   assert (ci(1), 0.5087487217143066, 1e-07);
%!   assert (ci(2), 0.9477093289635946, 1e-07);
%! end
%!
%! % Nonparametric 90% BCa confidence intervals (single bootstrap)
%! % BCa intervals on page 266 are 0.410 - 0.923
%! ci = bootci(1999,{@cor,LSAT,GPA},'alpha',0.1,'type','bca','seed',1);
%! if (isempty (regexp (which ('boot'), 'mex$')))
%!   % test boot m-file result
%!   assert (ci(1), 0.4132871532174259, 1e-07);
%!   assert (ci(2), 0.9237685885933666, 1e-07);
%! end
%!
%! % Nonparametric 90% calibrated percentile confidence intervals
%! % (double bootstrap)
%! ci = bootci(1999,{@cor,LSAT,GPA},'alpha',0.1,'type','cal', ...
%!                                                     'nbootcal',499,'seed',1);
%! if (isempty (regexp (which ('boot'), 'mex$')))
%!   % test boot m-file result
%!   assert (ci(1), 0.2899032097111454, 1e-07);
%!   assert (ci(2), 0.9347549589238046, 1e-07);
%! end
%! % Exact intervals based on normal theory are 0.51 - 0.91
