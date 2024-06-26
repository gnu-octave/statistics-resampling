% Performs balanced bootstrap (or bootknife) resampling of clusters or blocks of
% data and calculates bootstrap bias, standard errors and confidence intervals.
%
% -- Function File: bootclust (DATA)
% -- Function File: bootclust (DATA, NBOOT)
% -- Function File: bootclust (DATA, NBOOT, BOOTFUN)
% -- Function File: bootclust ({D1, D2, ...}, NBOOT, BOOTFUN)
% -- Function File: bootclust (DATA, NBOOT, {BOOTFUN, ...})
% -- Function File: bootclust (DATA, NBOOT, BOOTFUN, ALPHA)
% -- Function File: bootclust (DATA, NBOOT, BOOTFUN, ALPHA, CLUSTID)
% -- Function File: bootclust (DATA, NBOOT, BOOTFUN, ALPHA, BLOCKSZ)
% -- Function File: bootclust (DATA, NBOOT, BOOTFUN, ALPHA, ..., LOO)
% -- Function File: bootclust (DATA, NBOOT, BOOTFUN, ALPHA, ..., LOO, SEED)
% -- Function File: bootclust (DATA, NBOOT, BOOTFUN, ALPHA, ..., LOO, SEED, NPROC)
% -- Function File: STATS = bootclust (...)
% -- Function File: [STATS, BOOTSTAT] = bootclust (...)
%
%     'bootclust (DATA)' uses nonparametric balanced bootstrap resampling
%     to generate 1999 resamples from clusters or contiguous blocks of rows of
%     the DATA (column vector or matrix) [1]. By default, each row is it's own
%     cluster/block (i.e. no clustering or blocking). The means of the resamples
%     are then computed and the following statistics are displayed:
%        - original: the original estimate(s) calculated by BOOTFUN and the DATA
%        - bias: bootstrap estimate of the bias of the sampling distribution(s)
%        - std_error: bootstrap estimate(s) of the standard error(s)
%        - CI_lower: lower bound(s) of the 95% bootstrap confidence interval(s)
%        - CI_upper: upper bound(s) of the 95% bootstrap confidence interval(s)
%
%     'bootclust (DATA, NBOOT)' specifies the number of bootstrap resamples,
%     where NBOOT is a scalar, positive integer corresponding to the number
%     of bootstrap resamples. The default value of NBOOT is the scalar: 1999.
%
%     'bootclust (DATA, NBOOT, BOOTFUN)' also specifies BOOTFUN: the function
%     calculated on the original sample and the bootstrap resamples. BOOTFUN
%     must be either a:
%       <> function handle, function name or an anonymous function,
%       <> string of a function name, or
%       <> a cell array where the first cell is one of the above function
%          definitions and the remaining cells are (additional) input arguments 
%          to that function (after the data arguments).
%        In all cases BOOTFUN must take DATA for the initial input argument(s).
%        BOOTFUN can return a scalar or any multidimensional numeric variable,
%        but the output will be reshaped as a column vector. BOOTFUN must
%        calculate a statistic representative of the finite data sample; it
%        should NOT be an estimate of a population parameter (unless they are
%        one of the same). If BOOTFUN is @mean or 'mean', narrowness bias of
%        the confidence intervals for single bootstrap are reduced by expanding
%        the probabilities of the percentiles using Student's t-distribution
%        [2]. By default, BOOTFUN is @mean.
%
%     'bootclust ({D1, D2, ...}, NBOOT, BOOTFUN)' resamples from the clusters
%     or blocks of rows of the data vectors D1, D2 etc and the resamples are
%     passed onto BOOTFUN as multiple data input arguments. All data vectors
%     and matrices (D1, D2 etc) must have the same number of rows.
%
%     'bootclust (DATA, NBOOT, BOOTFUN, ALPHA)', where ALPHA is numeric
%     and sets the lower and upper bounds of the confidence interval(s). The
%     value(s) of ALPHA must be between 0 and 1. ALPHA can either be:
%       <> scalar: To set the (nominal) central coverage of equal-tailed
%                  percentile confidence intervals to 100*(1-ALPHA)%.
%       <> vector: A pair of probabilities defining the (nominal) lower and
%                  upper percentiles of the confidence interval(s) as
%                  100*(ALPHA(1))% and 100*(ALPHA(2))% respectively. The
%                  percentiles are bias-corrected and accelerated (BCa) [3].
%        The default value of ALPHA is the vector: [.025, .975], for a 95%
%        BCa confidence interval.
%
%     'bootclust (DATA, NBOOT, BOOTFUN, ALPHA, CLUSTID)' also sets CLUSTID,
%     which are identifiers that define the grouping of the DATA rows for
%     cluster bootstrap resampling. CLUSTID should be a column vector or
%     cell array with the same number of rows as the DATA. Rows in DATA with
%     the same CLUSTID value are treated as clusters of observations that are
%     resampled together.
%
%     'bootclust (DATA, NBOOT, BOOTFUN, ALPHA, BLOCKSZ)' groups consecutive
%     DATA rows into non-overlapping blocks of length BLOCKSZ for simple block
%     bootstrap resampling [4]. Note that this variation of block bootstrap is
%     a special case of resampling clustered data. By default, BLOCKSZ is 1.
%
%     'bootclust (DATA, NBOOT, BOOTFUN, ALPHA, ..., LOO)' sets the resampling
%     method. If LOO is false, the resampling method used is balanced bootstrap
%     resampling. If LOO is true, the resampling method used is balanced
%     bootknife resampling [5]. Where N is the number of clusters or blocks,
%     bootknife cluster or block resampling involves creating leave-one-out
%     jackknife samples of size N - 1, and then drawing resamples of size N with
%     replacement from the jackknife samples, thereby incorporating Bessel's
%     correction into the resampling procedure. LOO must be a scalar logical
%     value. The default value of LOO is false.
%
%     'bootclust (DATA, NBOOT, BOOTFUN, ALPHA, ..., LOO, SEED)' initialises
%     the Mersenne Twister random number generator using an integer SEED value
%     so that bootclust results are reproducible.
%
%     'bootclust (DATA, NBOOT, BOOTFUN, ALPHA, ..., LOO, SEED, NPROC)' also
%     sets the number of parallel processes to use for jackknife computations
%     and non-vectorized function evaluations during bootstrap and on multicore
%     machines. This feature requires the Parallel package (in Octave), or the
%     Parallel Computing Toolbox (in Matlab). This option is ignored during
%     bootstrap function evaluations when BOOTFUN is vectorized.
%
%     'STATS = bootclust (...)' returns a structure with the following fields
%     (defined above): original, bias, std_error, CI_lower, CI_upper.
%
%     '[STATS, BOOTSTAT] = bootclust (...)' returns BOOTSTAT, a vector or matrix
%     of bootstrap statistics calculated over the bootstrap resamples.
%
%     '[STATS, BOOTSTAT, BOOTDATA] = bootclust (...)' returns BOOTDATA, a 1-by-
%     NBOOT cell array of datasets generated by cluster or block bootstrap
%     resampling.
%
%  BIBLIOGRAPHY:
%  [1] Davison and Hinkley (1997). Bootstrap methods and their application
%        (Vol. 1). New York, NY: Cambridge University Press.
%  [2] Hesterberg, Tim (2014), What Teachers Should Know about the 
%        Bootstrap: Resampling in the Undergraduate Statistics Curriculum, 
%        http://arxiv.org/abs/1411.5279
%  [3] Efron and Tibshirani (1993) An Introduction to the Bootstrap. 
%        New York, NY: Chapman & Hall
%  [4] Carlstein (1986) The use of subseries values for estimating the
%        variance of a general statistic from a stationary sequence. 
%        Ann. Statist. 14, 1171-9
%  [5] Hesterberg (2004) Unbiasing the Bootstrap—Bootknife Sampling 
%        vs. Smoothing; Proceedings of the Section on Statistics & the 
%        Environment. Alexandria, VA: American Statistical Association.
%
%  bootclust (version 2024.05.16)
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

function [stats, bootstat, X] = bootclust (x, nboot, bootfun, alpha, ...
                                        clustid, loo, seed, ncpus)

  % Store subfunctions in a stucture to make them available for parallel processes
  parsubfun = struct ('col2args', @col2args, ...
                      'kdeinv', @kdeinv, ...
                      'ExpandProbs', @ExpandProbs);

  % Check if we are running Octave or Matlab
  info = ver; 
  ISOCTAVE = any (ismember ({info.Name}, 'Octave'));

  % Check the number of function arguments
  if (nargin < 1)
    error ('bootclust: DATA must be provided');
  end
  if (nargin > 8)
    error ('bootclust: Too many input arguments')
  end
  if (nargout > 3)
    error ('bootclust: Too many output arguments')
  end

  % NBOOT input argument
  if ((nargin < 2) || isempty (nboot))
    nboot = 1999;
  else
    if (~ isa (nboot, 'numeric'))
      error ('bootclust: NBOOT must be numeric');
    end
    if (numel (nboot) > 1)
      error ('bootclust: NBOOT cannot contain more than 1 value');
    end
    if (nboot ~= abs (fix (nboot)))
      error ('bootclust: NBOOT must contain positive integers');
    end    
  end
  if (~ all (size (nboot) == [1, 1]))
    error ('bootclust: NBOOT must be a scalar value')
  end

  % BOOTFUN input argument
  if ((nargin < 3) || isempty (bootfun))
    bootfun = @mean;
    bootfun_str = 'mean';
  else
    if (iscell (bootfun))
      if (ischar (bootfun{1}))
        % Convert character string of a function name to a function handle
        bootfun_str = bootfun{1};
        func = str2func (bootfun{1});
      else
        bootfun_str = func2str (bootfun{1});
        func = bootfun{1};
      end
      args = bootfun(2:end);
      bootfun = @(varargin) func (varargin{:}, args{:});
    elseif (ischar (bootfun))
      % Convert character string of a function name to a function handle
      bootfun_str = bootfun;
      bootfun = str2func (bootfun);
    elseif (isa (bootfun, 'function_handle'))
      bootfun_str = func2str (bootfun);
    else
      error ('bootclust: BOOTFUN must be a function name or function handle')
    end
  end

  % ALPHA input argument
  if ( (nargin < 4) || isempty (alpha) )
    alpha = [.025, .975];
  end
  nalpha = numel (alpha);
  if (~ isa (alpha, 'numeric') || (nalpha > 2))
    error (cat (2, 'bootclust: ALPHA must be a scalar (two-tailed', ...
                   'probability) or a vector (pair of probabilities)'))
  end
  if (size (alpha, 1) > 1)
    alpha = alpha.';
  end
  if (any (isnan (alpha)))
    error ('bootclust: ALPHA cannot contain NaN values');
  end
  if (any ((alpha < 0) | (alpha > 1)))
    error ('bootclust: Value(s) in ALPHA must be between 0 and 1');
  end
  if (nalpha > 1)
    % alpha is a pair of probabilities
    % Make sure probabilities are in the correct order
    if (alpha(1) > alpha(2) )
      error (cat (2, 'bootclust: The pair of probabilities must be', ...
                     ' in ascending numeric order'))
    end
    probs = alpha;
    alpha = 1 - probs(2) + probs(1);
  else
    probs = [alpha / 2 , 1 - alpha / 2];
  end

  % LOO input argument
  if ((nargin > 5) && ~ isempty (loo))
    if (~ islogical (loo))
      error ('bootclust: LOO must be a logical scalar value')
    end
  else
    loo = false;
  end
  
  % Initialise the random number generator with the SEED (if provided)
  if ( (nargin > 6) && (~ isempty (seed)) )
    boot (1, 1, false, seed);
  end

  % Evaluate NPROC input argument
  if ((nargin < 8) || isempty (ncpus)) 
    ncpus = 0;    % Ignore parallel processing features
  else
    if (~ isa (ncpus, 'numeric'))
      error ('bootclust: NPROC must be numeric');
    end
    if (any (ncpus ~= abs (fix (ncpus))))
      error ('bootclust: NPROC must be a positive integer');
    end    
    if (numel (ncpus) > 1)
      error ('bootclust: NPROC must be a scalar value');
    end
  end
  if (ISOCTAVE)
    ncpus = min (ncpus, nproc);
  else
    ncpus = min (ncpus, feature ('numcores'));
  end

  % If applicable, check we have parallel computing capabilities
  if (ncpus > 1)
    if (ISOCTAVE)
      software = pkg ('list');
      names = cellfun (@(S) S.name, software, 'UniformOutput', false);
      status = cellfun (@(S) S.loaded, software, 'UniformOutput', false);
      index = find (~ cellfun (@isempty, regexpi (names, '^parallel')));
      if (~ isempty (index))
        if (logical (status{index}))
          PARALLEL = true;
        else
          PARALLEL = false;
        end
      else
        PARALLEL = false;
      end
    else
      info = ver; 
      if (ismember ('Parallel Computing Toolbox', {info.Name}))
        PARALLEL = true;
      else
        PARALLEL = false;
      end
    end
  end

  % If applicable, setup a parallel pool (required for MATLAB)
  if (~ ISOCTAVE)
    % MATLAB
    % bootfun is not vectorized
    if (ncpus > 0) 
      % MANUAL
      try 
        pool = gcp ('nocreate'); 
        if isempty (pool)
          if (ncpus > 1)
            % Start parallel pool with ncpus workers
            parpool (ncpus);
          else
            % Parallel pool is not running and ncpus is 1 so run function
            % evaluations in serial
            ncpus = 1;
          end
        else
          if (pool.NumWorkers ~= ncpus)
            % Check if number of workers matches ncpus and correct it
            % accordingly if not
            delete (pool);
            if (ncpus > 1)
              parpool (ncpus);
            end
          end
        end
      catch
        % MATLAB Parallel Computing Toolbox is not installed
        warning ('bootknife:parallel', ...
                 cat (2, 'Parallel Computing Toolbox not installed or', ...
                         ' operational. Falling back to serial processing.'))
        ncpus = 1;
      end
    end
  else
    if ((ncpus > 1) && ~ PARALLEL)
      if (ISOCTAVE)
        % OCTAVE Parallel Computing Package is not installed or loaded
        warning ('bootknife:parallel', ...
                 cat (2, 'Parallel package is not installed and/or loaded.', ...
                         ' Falling back to serial processing.'))
      else
        % MATLAB Parallel Computing Toolbox is not installed or loaded
        warning ('bootknife:parallel', ...
                 cat (2, 'Parallel Computing Toolbox not installed and/or', ...
                         ' loaded. Falling back to serial processing.'))
      end
      ncpus = 0;
    end
  end

  % If DATA is a cell array of equal size colunmn vectors, convert the cell
  % array to a matrix and redefine bootfun to parse multiple input arguments
  if (iscell (x))
    szx = cellfun (@(x) size (x, 2), x);
    x = [x{:}];
    bootfun = @(x) parsubfun.col2args (bootfun, x, szx);
  else
    szx = size (x, 2);
  end

  % Determine properties of the DATA (x)
  [n, nvar] = size (x);
  if (n < 2)
    error ('bootclust: DATA must be numeric and contain > 1 row')
  end

  % Sort rows of CLUSTID and the DATA accordingly
  if ((nargin < 5) || isempty (clustid))
    clustid = (1 : n)';
    blocksz = 1;
  else
    if isscalar (clustid)
      % Group consecutive DATA rows into clusters of >= CLUSTID rows
      blocksz = clustid;
      if ( (~ isnumeric (blocksz)) || (blocksz ~= abs (blocksz)) || ...
                 (blocksz >= n) || (blocksz ~= fix (blocksz)) )
        error (cat (2, 'bootclust: BLOCKSZ must be a positive', ...
                       ' integer less than the number of DATA rows'))
      end
      nx = fix (n / blocksz);
      clustid = (nx + 1) * ones (n, 1);
      clustid(1:blocksz * nx, :) = reshape (ones (blocksz, 1) * (1:nx), [], 1);
      nx = clustid(end);
    else
      blocksz = [];
    end
    if ( any (size (clustid) ~= [n, 1]) )
      error (cat (2, 'bootclust: CLUSTID must be a column vector with', ...
                     ' the same number of rows as DATA'))
    end
    [clustid, idx] = sort (clustid);
    x = x(idx,:);
  end

  % Evaluate definition of the sampling units (e.g. clusters) of x 
  [ux, jnk, ic] = unique (clustid);
  nx = numel (ux);

  % Calculate the number of elements in the return value of bootfun and check
  % whether function evaluations can be vectorized
  T0 = bootfun (x);
  m = numel (T0);
  if (nvar > 1)
    M = cell2mat (cellfun (@(i) repmat (x(:, i), 1, 2), ... 
                 num2cell (1 : nvar), 'UniformOutput', false));
  else 
    M = repmat (x, 1, 2);
  end
  if (any (szx > 1))
    VECTORIZED = false;
  else
    try
      chk = bootfun (M);
      if (all (size (chk) == [size(T0, 1), 2]) && all (chk == bootfun (x)))
        VECTORIZED = true;
      else
        VECTORIZED = false;
      end
    catch
      VECTORIZED = false;
    end
  end
  if (m > 1)
    % Vectorized along the dimension of the return values of bootfun so
    % reshape the output to be a column vector before proceeding with bootstrap
    if (size (T0, 2) > 1)
      bootfun = @(x) reshape (bootfun (x), [], 1);
      T0 = reshape (T0, [], 1);
      VECTORIZED = false;
    end
  end
  % Check if we can vectorize function evaluations
  if (any (diff (accumarray (ic, 1))))
    VECTORIZED = false;
  end

  % Convert x to a cell array of clusters
  x = mat2cell (x, accumarray (ic, 1));

  % Perform resampling of clusters
  bootsam = boot (nx, nboot, loo);
  X = arrayfun (@(b) cell2mat (x(bootsam(:, b))), 1 : nboot, ...
                     'UniformOutput', false);

  % Perform the function evaluations
  if (VECTORIZED)
    if (nvar > 1)
      % Multivariate
      bootstat = bootfun (cell2mat (mat2cell (reshape (cell2mat (X), ...
                          n * nvar, nboot), repmat (n, nvar, 1))'));
    else
      % Univariate
      bootstat = bootfun (cell2mat (X));
    end
  else
    if (ncpus > 1)
      % Evaluate bootfun on each bootstrap resample in PARALLEL
      if (ISOCTAVE)
        % OCTAVE
        bootstat = parcellfun (ncpus, @(x) bootfun (x), X, ...
                               'UniformOutput', false);
      else
        % MATLAB
        bootstat = cell (1, nboot);
        parfor b = 1 : nboot; bootstat{b} = bootfun (X{:, b}); end
      end
    else
      % Evaluate bootfun on each bootstrap resample in SERIAL
      bootstat = cell2mat (arrayfun (@(b) bootfun (X{:, b}), ...
                                          1 : nboot, 'UniformOutput', false));
    end
  end
  if (iscell (bootstat))
    bootstat = cell2mat (bootstat);
  end

  % Remove bootstrap statistics that contain NaN or inf
  ridx = any (or (isnan (bootstat), isinf (bootstat)) , 1);
  bootstat(:, ridx) = [];
  if (isempty (bootstat))
    error ('bootclust: BOOTFUN returned NaN or inf for all bootstrap resamples')
  end
  nboot = nboot - sum (ridx);

  % Bootstrap bias estimation
  bias = mean (bootstat, 2) - T0;

  % Bootstrap standard error
  se = std (bootstat, 0, 2);

  % Make corrections to the probabilities for the lower and upper bounds of the
  % confidence intervals.
  % First, if bootfun is the arithmetic meam, expand the probabilities of the 
  % percentiles for the confidence intervals using Student's t-distribution
  if (strcmpi (bootfun_str, 'mean'))
    probs = parsubfun.ExpandProbs (probs, nx - 1, loo);
  end
  % If requested, perform adjustments to the probabilities to correct for bias
  % and skewness
  switch (nalpha)
    case 1
      % No adjustments made
      probs = repmat (probs, m, 1);
    case 2
      % Create distribution functions
      stdnormcdf = @(x) 0.5 * (1 + erf (x / sqrt (2)));
      stdnorminv = @(p) sqrt (2) * erfinv (2 * p - 1);
      % Try using Jackknife resampling to calculate the acceleration constant (a)
      state = warning;
      if (ISOCTAVE)
        warning ('on', 'quiet');
      else
        warning ('off', 'all');
      end
      try
        if (VECTORIZED)
          % Leave-one-out DATA resampling followed by vectorized function
          % evaluations
          if (nvar > 1)
            % Multivariate
            T = bootfun (reshape (cell2mat (arrayfun (...
                         @(i) vertcat (x{1 : nx ~= i, :}), (1 : nx)', ...
                         'UniformOutput', false)), n - n / nx, []));
          else
            % Univariate
            T = bootfun (cell2mat (arrayfun (...
                         @(i) vertcat (x{1 : nx ~= i, :}), (1 : nx), ...
                         'UniformOutput', false)));
          end
        else
          % Leave-one-out DATA resampling followed by looped function
          % evaluations (if bootfun is not vectorized)
          jackfun = @(i) bootfun (vertcat (x{1 : nx ~= i, :}));
          if (ncpus > 1)  
            % PARALLEL evaluation of bootfun on each jackknife resample 
            if (ISOCTAVE)
              % OCTAVE
              T = cell2mat (pararrayfun (ncpus, jackfun, 1 : nx, ...
                                         'UniformOutput', false));
            else
              % MATLAB
              T = zeros (m, nx);
              parfor i = 1 : nx; T(:, i) = feval (jackfun, i); end
            end
          else
            % SERIAL evaluation of bootfun on each jackknife resample
            T = cell2mat (arrayfun (jackfun, 1 : nx, 'UniformOutput', false));
          end
        end
        % Calculate empirical influence function
        U = (nx - 1) * bsxfun (@minus, T0, T);
        a = sum (U.^3, 2) ./ (6 * sum (U.^2, 2) .^ 1.5);
      catch
        % Revert to bias-corrected (BC) bootstrap confidence intervals
        warning ('bootclust:jackfail', cat (2, 'BOOTFUN failed during', ... 
              ' jackknife calculations; acceleration constant set to 0.\n'))
        a = zeros (m, 1);
      end
      % Calculate the median bias correction constant (z0)
      z0 = stdnorminv (sum (bsxfun (@lt, bootstat, T0), 2) / nboot);
      if (~ all (isfinite (z0)))
        % Revert to percentile bootstrap confidence intervals
        warning ('bootclust:biasfail', ...
                 cat (2, 'Unable to calculate the bias correction', ...
                         ' constant; reverting to percentile intervals.\n'))
        z0 = zeros (m, 1);
        a = zeros (m, 1);
      end
      % Calculate BCa or BC percentiles
      z = stdnorminv (probs);
      probs = stdnormcdf (bsxfun (@plus, z0, bsxfun (@plus, z0, z) ./ ...
                          (1 - (bsxfun (@times, a, bsxfun (@plus, z0, z))))));
  end

  % Intervals constructed from kernel density estimate of the bootstrap
  % statistics (with shrinkage correction)
  ci = nan (m, 2);
  for j = 1 : m
    try
      ci(j, :) = parsubfun.kdeinv (probs(j, :), bootstat(j, :), ...
                         se(j) * sqrt (1 / (nx - 1)), 1 - 1 / (nx - 1));
    catch
      % Linear interpolation (legacy)
      fprintf (strcat ('Note: Falling back to linear interpolation to', ...
                       ' calculate percentiles for interval pair %u\n'), j);
      [t1, cdf] = bootcdf (bootstat(j, :), true, 1);
      ci(j, 1) = interp1 (cdf, t1, probs(1), 'linear', min (t1));
      ci(j, 2) = interp1 (cdf, t1, probs(2), 'linear', max (t1));
    end
  end

  % Create STATS output structure
  stats = struct;
  stats.original = T0;
  stats.bias = bias;          % Bootstrap bias estimation
  stats.std_error = se;       % Bootstrap standard error
  stats.CI_lower = ci(:, 1);  % Lower percentile
  stats.CI_upper = ci(:, 2);  % Upper percentile

  % Print output if no output arguments are requested
  if (nargout == 0) 
    print_output (stats, nboot, nalpha, alpha, probs, m, bootfun_str, ...
                  loo, blocksz);
  else
    if (isempty (bootsam))
      [warnmsg, warnID] = lastwarn;
      if (ismember (warnID, {'bootclust:biasfail','bootclust:jackfail'}))
        warning ('bootclust:lastwarn', warnmsg);
      end
      lastwarn ('', '');
    end
  end

end

%--------------------------------------------------------------------------

function retval = col2args (func, x, szx)

  % Usage: retval = col2args (func, x, nvar)
  % col2args evaluates func on the columns of x. When nvar > 1, each of the
  % blocks of x are passed to func as a separate arguments. 

  % Extract columns of the matrix into a cell array
  [n, ncols] = size (x);
  xcell = mat2cell (x, n, ncols / sum (szx) * szx);

  % Evaluate column vectors as independent of arguments to bootfun
  retval = func (xcell{:});

end

%--------------------------------------------------------------------------

function X = kdeinv (P, Y, BW, CF)

  % Inverse of the cumulative density function (CDF) of a kernel density 
  % estimate (KDE)
  % 
  % The function returns X, the inverse CDF of the KDE of Y for the bandwidth
  % BW evaluated at the values in P. CF is a shrinkage factor for the variance
  % of the data in Y

  % Set defaults for optional input arguments
  if (nargin < 4)
    CF = 1;
  end

  % Create Normal CDF function
  pnorm = @(X, MU, SD) (0.5 * (1 + erf ((X - MU) / (SD * sqrt (2)))));

  % Calculate statistics of the data
  N = numel (Y);
  MU = mean (Y);

  % Apply shrinkage correction
  Y = ((Y - MU) * sqrt (CF)) + MU;

  % Set initial values of X0
  YS = sort (Y, 2);
  X0 = YS(fix ((N - 1) * P) + 1);

  % Perform root finding to get quantiles of the KDE at values of P
  findroot = @(X0, P) fzero (@(X) sum (pnorm (X - Y, 0, BW)) / N - P, X0);
  X = [-Inf, +Inf];
  for i = 1 : numel(P)
    if (~ ismember (P(i), [0, 1]))
      X(i) = findroot (X0(i), P(i));
    end
  end

end

%--------------------------------------------------------------------------

function PX = ExpandProbs (P, DF, LOO)

  % Modify ALPHA to adjust tail probabilities assuming that the kurtosis
  % of the sampling distribution scales with degrees of freedom like the
  % t-distribution. This is related in concept to ExpandProbs in the
  % R package 'resample':
  % www.rdocumentation.org/packages/resample/versions/0.6/topics/ExpandProbs

  % Get size of P
  sz = size (P);

  % Create required distribution functions
  stdnormcdf = @(X) 0.5 * (1 + erf (X / sqrt (2)));
  stdnorminv = @(P) sqrt (2) * erfinv (2 * P - 1);
  if ((exist ('betaincinv', 'builtin')) || (exist ('betaincinv', 'file')))
    studinv = @(P, DF) sign (P - 0.5) * ...
                sqrt ( DF ./ betaincinv (2 * min (P, 1 - P), DF / 2, 0.5) - DF);
  else
    % Earlier versions of Matlab do not have betaincinv
    % Instead, use betainv from the Statistics and Machine Learning Toolbox
    try 
      studinv = @(P, DF) sign (P - 0.5) * ...
                  sqrt ( DF ./ betainv (2 * min (P, 1 - P), DF / 2, 0.5) - DF);
    catch
      % Use the Normal distribution (i.e. do not expand probabilities) if
      % either betaincinv or betainv are not available
      studinv = @(P, DF) stdnorminv (P);
      warning ('bootclust:ExpandProbs', ...
          'Could not create studinv function; intervals will not be expanded.');
    end
  end
 
  % Calculate expanded probabilities
  if LOO
    PX = stdnormcdf (arrayfun (studinv, P, repmat (DF, sz)));
  else
    n = DF + 1;
    PX = stdnormcdf (sqrt (n / (n - 1)) * ...
                     arrayfun (studinv, P, repmat (DF, sz)));
  end

end

%--------------------------------------------------------------------------

function print_output (stats, nboot, nalpha, alpha, probs, m, bootfun_str, ...
                       loo, blocksz)

    if (isempty (blocksz))
      bootname = 'cluster';
    else
      bootname = 'block';
    end
    fprintf (cat (2, '\nSummary of nonparametric %s bootstrap', ...
                     ' estimates of bias and precision\n', ...
                     '*************************************************', ...
                     '*****************************\n\n'), bootname);
    fprintf ('Bootstrap settings: \n');
    fprintf (' Function: %s\n', bootfun_str);
    if loo
      fprintf (' Resampling method: Balanced, %s bootknife resampling \n', ...
               bootname);
    else
      fprintf (' Resampling method: Balanced, %s bootstrap resampling \n', ...
               bootname);
    end
    fprintf (' Number of resamples: %u \n', nboot(1));
    if (~ isempty (blocksz))
      fprintf (' Number of data rows in each block: %u \n', blocksz);
    end
    if (nalpha > 1)
      [jnk, warnID] = lastwarn;
      switch warnID
        case 'bootclust:biasfail'
          if (strcmpi (bootfun_str, 'mean'))
            fprintf (cat (2, ' Confidence interval (CI) type:', ...
                             ' Expanded percentile\n'));
          else
            fprintf (' Confidence interval (CI) type: Percentile\n');
          end
        case 'bootclust:jackfail'
          if (strcmpi (bootfun_str, 'mean'))
            fprintf (cat (2, ' Confidence interval (CI) type:', ...
                             ' Expanded bias-corrected (BC) \n'));
          else
            fprintf (cat (2, ' Confidence interval (CI) type:', ...
                             ' Bias-corrected (BC) \n'));
          end
        otherwise
          if (strcmpi (bootfun_str, 'mean'))
            fprintf (cat (2, ' Confidence interval (CI) type: Expanded', ...
                             ' bias-corrected and accelerated (BCa) \n'));
          else
            fprintf (cat (2, ' Confidence interval (CI) type: Bias-', ...
                             'corrected and accelerated (BCa) \n'));
          end
      end
    else
      if (strcmpi (bootfun_str, 'mean'))
        fprintf (cat (2, ' Confidence interval (CI) type: Expanded', ...
                         ' percentile (equal-tailed)\n'));
      else
        fprintf (cat (2, ' Confidence interval (CI) type: Percentile', ...
                         ' (equal-tailed)\n'));
      end
    end
    coverage = 100 * (1 - alpha);
    if (all (bsxfun (@eq, probs, probs(1, :))))
      fprintf (cat (2, ' Nominal coverage (and the percentiles used):', ...
                       ' %.3g%% (%.1f%%, %.1f%%)\n\n'), ...
                       coverage, 100 * probs(1,:));
    else
      fprintf (' Nominal coverage: %.3g%%\n\n', coverage);
    end
    fprintf ('Bootstrap Statistics: \n');
    fprintf (cat (2, ' original     bias         std_error    CI_lower', ...
                     '     CI_upper  \n'));
    for i = 1 : m
      fprintf (cat (2, ' %#-+10.4g   %#-+10.4g   %#-+10.4g   %#-+10.4g', ...
                     '   %#-+10.4g \n'), [stats.original(i), stats.bias(i), ...
                     stats.std_error(i), stats.CI_lower(i), stats.CI_upper(i)]);
    end
    fprintf ('\n');
    lastwarn ('', '');  % reset last warning

end

%--------------------------------------------------------------------------

%!demo
%!
%! % Input univariate dataset
%! data = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%!         0 33 28 34 4 32 24 47 41 24 26 30 41].';
%!
%! % 95% expanded BCa bootstrap confidence intervals for the mean
%! bootclust (data, 1999, @mean);
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input univariate dataset
%! data = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%!         0 33 28 34 4 32 24 47 41 24 26 30 41].';
%! clustid = {'a';'a';'b';'b';'a';'c';'c';'d';'e';'e';'e';'f';'f'; ...
%!            'g';'g';'g';'h';'h';'i';'i';'j';'j';'k';'l';'m';'m'};
%!
%! % 95% expanded BCa bootstrap confidence intervals for the mean with
%! % cluster resampling
%! bootclust (data, 1999, @mean, [0.025,0.975], clustid);
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input univariate dataset
%! data = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%!         0 33 28 34 4 32 24 47 41 24 26 30 41].';
%!
%! % 90% equal-tailed percentile bootstrap confidence intervals for
%! % the variance
%! bootclust (data, 1999, {@var, 1}, 0.1);
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input univariate dataset
%! data = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%!         0 33 28 34 4 32 24 47 41 24 26 30 41].';
%! clustid = {'a';'a';'b';'b';'a';'c';'c';'d';'e';'e';'e';'f';'f'; ...
%!            'g';'g';'g';'h';'h';'i';'i';'j';'j';'k';'l';'m';'m'};
%!
%! % 90% equal-tailed percentile bootstrap confidence intervals for
%! % the variance
%! bootclust (data, 1999, {@var, 1}, 0.1, clustid);
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input univariate dataset
%! data = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%!         0 33 28 34 4 32 24 47 41 24 26 30 41].';
%!
%! % 90% BCa bootstrap confidence intervals for the variance
%! bootclust (data, 1999, {@var, 1}, [0.05 0.95]);
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input univariate dataset
%! data = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%!         0 33 28 34 4 32 24 47 41 24 26 30 41].';
%! clustid = {'a';'a';'b';'b';'a';'c';'c';'d';'e';'e';'e';'f';'f'; ...
%!            'g';'g';'g';'h';'h';'i';'i';'j';'j';'k';'l';'m';'m'};
%!
%! % 90% BCa bootstrap confidence intervals for the variance
%! bootclust (data, 1999, {@var, 1}, [0.05 0.95], clustid);
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input dataset
%! y = randn (20,1); x = randn (20,1); X = [ones(20,1), x];
%!
%! % 90% BCa confidence interval for regression coefficients 
%! bootclust ({X,y}, 1999, @mldivide, [0.05 0.95]);
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input dataset
%! y = randn (20,1); x = randn (20,1); X = [ones(20,1), x];
%! clustid = [1;1;1;1;2;2;2;3;3;3;3;4;4;4;4;4;5;5;5;6];
%!
%! % 90% BCa confidence interval for regression coefficients 
%! bootclust ({X,y}, 1999, @mldivide, [0.05 0.95], clustid);
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Input bivariate dataset
%! x = [576 635 558 578 666 580 555 661 651 605 653 575 545 572 594].';
%! y = [3.39 3.3 2.81 3.03 3.44 3.07 3 3.43 ...
%!      3.36 3.13 3.12 2.74 2.76 2.88 2.96].';
%! clustid = [1;1;3;1;1;2;2;2;2;3;1;3;3;3;2];
%!
%! % 95% BCa bootstrap confidence intervals for the correlation coefficient
%! bootclust ({x, y}, 1999, @cor, [], clustid);
%!
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % The following dataset represents lutenizing hormone levels measured in a
%! % healthy women every 10 minutes over an 8 hour period. The dataset was the
%! % example tabulated on page 92 of Efron and Tibshirani (1993) An Introduction
%! % to the Bootstrap.
%! y=[2.4;2.4;2.4;2.2;2.1;1.5;2.3;2.3; 2.5;2.0;1.9;1.7;2.2;1.8;3.2;3.2;...
%!    2.7;2.2;2.2;1.9;1.9;1.8;2.7;3.0;2.3;2.0;2.0;2.9;2.9;2.7;2.7;2.3;...
%!    2.6;2.4;1.8;1.7;1.5;1.4;2.1;3.3;3.5;3.5;3.1;2.6;2.1;3.4;3.0;2.9];
%!
%! % Calculation of the standardized lutenizing hormone levels is as follows
%! z = y - mean(y);
%!
%! % Let us then calculate the coefficient for a first order autoregressive
%! % model, AR(1), which can be used to make future predictions of the level
%! % of lutenizing hormone from the last measurement taken. We will use block
%! % bootstrap using a block size of 3 to obtain an estimate of the standard
%! % error and 95% confidence intervals around the regression coefficient
%! % estimate.
%! betafunc = @(y) (y(1:end-1) - mean(y)) \ (y(2:end) - mean(y));
%! blocksz = 3;
%! seed = 2;
%! bootclust(y,1999,betafunc,[0.025,0.975],blocksz,true,seed);
%!
%! % The estimate of beta here is 0.586 and the standard error is about 0.13.
%! % The coefficient indicates that we can predict that standardized hormone
%! % levels to change by a factor of 0.586 from the previous timepoint.

%!test
%! % Test for errors when using different functionalities of bootclust
%! y = randn (20,1); 
%! clustid = [1;1;1;1;1;1;1;1;1;1;2;2;2;2;2;3;3;3;3;3];
%! stats = bootclust (y, 1999, @mean);
%! stats = bootclust (y, 1999, 'mean');
%! stats = bootclust (y, 1999, {@var,1});
%! stats = bootclust (y, 1999, {'var',1});
%! stats = bootclust (y, 1999, @mean, [], 4);
%! stats = bootclust (y, 1999, @mean, [], clustid);
%! stats = bootclust (y, 1999, {'var',1}, [], clustid);
%! stats = bootclust (y, 1999, {'var',1}, [], clustid, true);
%! stats = bootclust (y, 1999, {@var,1}, [], clustid, true, 1);
%! stats = bootclust (y, 1999, @mean, .1, clustid, true);
%! stats = bootclust (y, 1999, @mean, .1, clustid, true, 1);
%! stats = bootclust (y, 1999, @mean, [.05,.95], clustid, true);
%! stats = bootclust (y, 1999, @mean, [.05,.95], clustid, true, 1);
%! stats = bootclust (y(1:5), 1999, @mean, .1);
%! stats = bootclust (y(1:5), 1999, @mean, [.05,.95]);
%! Y = randn (20); 
%! clustid = [1;1;1;1;1;1;1;1;1;1;2;2;2;2;2;3;3;3;3;3];
%! stats = bootclust (Y, 1999, @mean);
%! stats = bootclust (Y, 1999, 'mean');
%! stats = bootclust (Y, 1999, {@var, 1});
%! stats = bootclust (Y, 1999, {'var',1});
%! stats = bootclust (Y, 1999, @mean, [], clustid);
%! stats = bootclust (Y, 1999, {'var',1}, [], clustid);
%! stats = bootclust (Y, 1999, {@var,1}, [], clustid, true);
%! stats = bootclust (Y, 1999, {@var,1}, [], clustid, true, 1);
%! stats = bootclust (Y, 1999, @mean, .1, clustid, true);
%! stats = bootclust (Y, 1999, @mean, .1, clustid, true, 1);
%! stats = bootclust (Y, 1999, @mean, [.05,.95], clustid, true);
%! stats = bootclust (Y, 1999, @mean, [.05,.95], clustid, true, 1);
%! stats = bootclust (Y(1:5,:), 1999, @mean, .1);
%! stats = bootclust (Y(1:5,:), 1999, @mean, [.05,.95]);
%! y = randn (20,1); x = randn (20,1); X = [ones(20,1), x];
%! stats = bootclust ({x,y}, 1999, @cor);
%! stats = bootclust ({x,y}, 1999, @cor, [], clustid);
%! stats = bootclust ({x,y}, 1999, @mldivide);
%! stats = bootclust ({X,y}, 1999, @mldivide);
%! stats = bootclust ({X,y}, 1999, @mldivide, [], clustid);
%! stats = bootclust ({X,y}, 1999, @mldivide, [], clustid, true);
%! stats = bootclust ({X,y}, 1999, @mldivide, [], clustid, true, 1);
%! stats = bootclust ({X,y}, 1999, @mldivide, [.05,.95], clustid);

%!test
%! % Air conditioning failure times in Table 1.2 of Davison A.C. and
%! % Hinkley D.V (1997) Bootstrap Methods And Their Application. (Cambridge
%! % University Press)
%! x = [3, 5, 7, 18, 43, 85, 91, 98, 100, 130, 230, 487]';
%!
%! % Nonparametric 95% expanded percentile confidence intervals (equal-tailed)
%! % Balanced bootknife resampling
%! % Example 5.4 percentile intervals are 43.9 - 192.1
%! % Note that the intervals calculated below are wider because the narrowness
%! % bias was removed by expanding the probabilities of the percentiles using
%! % Student's t-distribution
%! stats = bootclust(x,1999,@mean,0.05,[],true,1);
%! if (isempty (regexp (which ('boot'), 'mex$')))
%!   % test boot m-file result
%!   assert (stats.original, 108.0833333333333, 1e-08);
%!   assert (stats.bias, 8.526512829121202e-14, 1e-08);
%!   assert (stats.std_error, 39.13480972156448, 1e-08);
%!   assert (stats.CI_lower, 35.93016716402185, 1e-08);
%!   assert (stats.CI_upper, 202.9345038615351, 1e-08);
%! end
%!
%! % Nonparametric 95% expanded BCa confidence intervals
%! % Balanced bootknife resampling
%! % Example 5.8 BCa intervals are 55.33 - 243.5
%! % Note that the intervals calculated below are wider because the narrowness
%! % bias was removed by expanding the probabilities of the percentiles using
%! % Student's t-distribution
%! stats = bootclust(x,1999,@mean,[0.025,0.975],[],true,1);
%! if (isempty (regexp (which ('boot'), 'mex$')))
%!   % test boot m-file result
%!   assert (stats.original, 108.0833333333333, 1e-08);
%!   assert (stats.bias, 8.526512829121202e-14, 1e-08);
%!   assert (stats.std_error, 39.13480972156448, 1e-08);
%!   assert (stats.CI_lower, 48.80198765188484, 1e-08);
%!   assert (stats.CI_upper, 246.531254466907, 1e-08);
%! end
%!
%! % Exact intervals based on an exponential model are 65.9 - 209.2
%! % (Example 2.11)

%!test
%! % Spatial test data from Table 14.1 of Efron and Tibshirani (1993)
%! % An Introduction to the Bootstrap in Monographs on Statistics and Applied 
%! % Probability 57 (Springer)
%! A = [48 36 20 29 42 42 20 42 22 41 45 14 6 ...
%!      0 33 28 34 4 32 24 47 41 24 26 30 41]';
%!
%! % Nonparametric 90% equal-tailed percentile confidence intervals
%! % Balanced bootknife resampling
%! % Table 14.2 percentile intervals are 100.8 - 233.9
%! stats = bootclust(A,1999,{@var,1},0.1,[],true,1);
%! if (isempty (regexp (which ('boot'), 'mex$')))
%!   % test boot m-file result
%!   assert (stats.original, 171.534023668639, 1e-08);
%!   assert (stats.bias, -6.916841556872669, 1e-08);
%!   assert (stats.std_error, 42.5668171689963, 1e-08);
%!   assert (stats.CI_lower, 96.28173295410532, 1e-08);
%!   assert (stats.CI_upper, 236.5630928698122, 1e-08);
%! end
%!
%! % Nonparametric 90% BCa confidence intervals
%! % Balanced bootknife resampling
%! % Table 14.2 BCa intervals are 115.8 - 259.6
%! stats = bootclust(A,1999,{@var,1},[0.05 0.95],[],true,1);
%! if (isempty (regexp (which ('boot'), 'mex$')))
%!   % test boot m-file result
%!   assert (stats.original, 171.534023668639, 1e-08);
%!   assert (stats.bias, -6.916841556872669, 1e-08);
%!   assert (stats.std_error, 42.5668171689963, 1e-08);
%!   assert (stats.CI_lower, 117.211853429442, 1e-08);
%!   assert (stats.CI_upper, 270.1620774419748, 1e-08);
%! end
%!
%! % Exact intervals based on normal theory are 118.4 - 305.2 (Table 14.2)
%! % Note that all of the bootknife intervals are slightly wider than the
%! % nonparametric intervals in Table 14.2 because the bootknife (rather than
%! % standard bootstrap) resampling used here reduces small sample bias

%!test
%! % Law school data from Table 3.1 of Efron and Tibshirani (1993)
%! % An Introduction to the Bootstrap in Monographs on Statistics and Applied 
%! % Probability 57 (Springer)
%! LSAT = [576 635 558 578 666 580 555 661 651 605 653 575 545 572 594]';
%! GPA = [3.39 3.3 2.81 3.03 3.44 3.07 3 3.43 ...
%!        3.36 3.13 3.12 2.74 2.76 2.88 2.96]';
%!
%! % Nonparametric 90% equal-tailed percentile confidence intervals
%! % Balanced bootstrap resampling
%! % Percentile intervals on page 266 are 0.524 - 0.928
%! stats = bootclust({LSAT,GPA},1999,@cor,0.1,[],false,1);
%! if (isempty (regexp (which ('boot'), 'mex$')))
%!   % test boot m-file result
%!   assert (stats.original, 0.7763744912894071, 1e-08);
%!   assert (stats.bias, -0.007472442917410338, 1e-08);
%!   assert (stats.std_error, 0.1355235987517371, 1e-08);
%!   assert (stats.CI_lower, 0.5147126713756366, 1e-08);
%!   assert (stats.CI_upper, 0.953156157402217, 1e-08);
%! end
%!
%! % Nonparametric 90% BCa confidence intervals
%! % Balanced bootstrap resampling
%! % BCa intervals on page 266 are 0.410 - 0.923
%! stats = bootclust({LSAT,GPA},1999,@cor,[0.05 0.95],[],false,1);
%! if (isempty (regexp (which ('boot'), 'mex$')))
%!   % test boot m-file result
%!   assert (stats.original, 0.7763744912894071, 1e-08);
%!   assert (stats.bias, -0.007472442917410338, 1e-08);
%!   assert (stats.std_error, 0.1355235987517371, 1e-08);
%!   assert (stats.CI_lower, 0.4193007209659811, 1e-08);
%!   assert (stats.CI_upper, 0.9242671615802232, 1e-08);
%! end
