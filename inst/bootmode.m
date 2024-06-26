% Uses bootstrap to evaluate the likely number of real peaks (i.e. modes)
% in the distribution of a single set of data.
%
% -- Function File: H = bootmode (X, M)
% -- Function File: H = bootmode (X, M, NBOOT)
% -- Function File: H = bootmode (X, M, NBOOT, KERNEL)
% -- Function File: H = bootmode (X, M, NBOOT, KERNEL, NPROC)
% -- Function File: [H, P] = bootmode (X, M, ...)
% -- Function File: [H, P, CRITVAL] = bootmode (X, M, ...)
%
%     'H = bootmode (X, M)' tests whether the distribution underlying the 
%     univariate data in vector X has M modes. The method employs the
%     smooth bootstrap as described [1]. The parsimonious approach is to
%     iteratively call this function, each time incrementally increasing
%     the number of modes until the null hypothesis (H0) is accepted (i.e.
%     H=0), where H0 corresponds to the number of modes being equal to M. 
%        - If H = 0, H0 cannot be rejected at the 5% significance level.
%        - If H = 1, H0 can be rejected at the 5% significance level.
%
%     'H = bootmode (X, M, NBOOT)' sets the number of bootstrap replicates
%
%     'H = bootmode (X, M, NBOOT, KERNEL)' sets the kernel for kernel
%     density estimation. Possible values are:
%        o 'Gaussian' (default)
%        o 'Epanechnikov'
%
%     'H = bootmode (X, M, NBOOT, KERNEL, NPROC)' sets the number of parallel
%      processes to use to accelerate computations. This feature requires the
%      Parallel package (in Octave), or the Parallel Computing Toolbox (in
%      Matlab).
%
%     '[H, P] = bootmode (X, M, ...)' also returns the two-tailed p-value of
%      the bootstrap hypothesis test.
%
%     '[H, P, CRITVAL] = bootmode (X, M, ...)' also returns the critical
%     bandwidth (i.e.the smallest bandwidth achievable to obtain a kernel
%     density estimate with M modes)
%
%  Bibliography:
%  [1] Efron and Tibshirani. Chapter 16 Hypothesis testing with the
%       bootstrap in An introduction to the bootstrap (CRC Press, 1994)
%
%  bootmode (version 2023.05.02)
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


% Stamp data example used in reference [1] in bootstrap R package
% x=[0.060;0.064;0.064;0.065;0.066;0.068;0.069;0.069;0.069;0.069;0.069; ...
%    0.069;0.069;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070; ...
%    0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070; ...
%    0.070;0.070;0.070;0.070;0.070;0.070;0.071;0.071;0.071;0.071;0.071; ...
%    0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071; ...
%    0.071;0.071;0.071;0.071;0.072;0.072;0.072;0.072;0.072;0.072;0.072; ...
%    0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072; ...
%    0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072; ...
%    0.072;0.072;0.072;0.073;0.073;0.073;0.073;0.073;0.073;0.073;0.073; ...
%    0.073;0.073;0.073;0.074;0.074;0.074;0.074;0.074;0.074;0.074;0.074; ...
%    0.074;0.074;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075; ...
%    0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075; ...
%    0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076; ...
%    0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.077;0.077;0.077;0.077; ...
%    0.077;0.077;0.077;0.077;0.077;0.077;0.077;0.078;0.078;0.078;0.078; ...
%    0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078; ...
%    0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.079;0.079;0.079; ...
%    0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079; ...
%    0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079; ...
%    0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079; ...
%    0.079;0.079;0.079;0.079;0.079;0.079;0.080;0.080;0.080;0.080;0.080; ...
%    0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080; ...
%    0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080; ...
%    0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.081; ...
%    0.081;0.081;0.081;0.081;0.081;0.081;0.081;0.081;0.081;0.081;0.081; ...
%    0.081;0.081;0.081;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082; ...
%    0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.083; ...
%    0.083;0.083;0.083;0.083;0.083;0.083;0.084;0.084;0.084;0.085;0.085; ...
%    0.086;0.086;0.087;0.088;0.088;0.089;0.089;0.089;0.089;0.089;0.089; ...
%    0.089;0.089;0.089;0.089;0.090;0.090;0.090;0.090;0.090;0.090;0.090; ...
%    0.090;0.090;0.091;0.091;0.091;0.092;0.092;0.092;0.092;0.092;0.093; ...
%    0.093;0.093;0.093;0.093;0.093;0.094;0.094;0.094;0.095;0.095;0.096; ...
%    0.096;0.096;0.097;0.097;0.097;0.097;0.097;0.097;0.097;0.098;0.098; ...
%    0.098;0.098;0.098;0.099;0.099;0.099;0.099;0.099;0.100;0.100;0.100; ...
%    0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100; ...
%    0.100;0.101;0.101;0.101;0.101;0.101;0.101;0.101;0.101;0.101;0.102; ...
%    0.102;0.102;0.102;0.102;0.102;0.102;0.102;0.103;0.103;0.103;0.103; ...
%    0.103;0.103;0.103;0.104;0.104;0.105;0.105;0.105;0.105;0.105;0.106; ...
%    0.106;0.106;0.106;0.107;0.107;0.107;0.108;0.108;0.108;0.108;0.108; ...
%    0.108;0.108;0.109;0.109;0.109;0.109;0.109;0.109;0.109;0.110;0.110; ...
%    0.110;0.110;0.110;0.110;0.110;0.110;0.110;0.110;0.110;0.111;0.111; ...
%    0.111;0.111;0.112;0.112;0.112;0.112;0.112;0.114;0.114;0.114;0.115; ...
%    0.115;0.115;0.117;0.119;0.119;0.119;0.119;0.120;0.120;0.120;0.121; ...
%    0.122;0.122;0.123;0.123;0.125;0.125;0.128; 0.129;0.129;0.129;0.130;0.131]

% Results from multimodality testing using this
% function with the Gaussian kernel compare quantitatively with reference [1]
% Inference is similar when using an Epanechnikov kernel
%
% [H, P, h] = bootmode(x,m,2000,'Gaussian')
% m  h       P     H
% 1  0.0068  0.00  1
% 2  0.0032  0.32  0 <- smallest m where H0 accepted
% 3  0.0030  0.07  0
% 4  0.0028  0.00  1
% 5  0.0026  0.00  1
% 6  0.0024  0.00  1
% 7  0.0015  0.46  0
%
% [H, P, h] = bootmode(x,m,2000,'Epanechnikov')
% m  h       P     H
% 1  0.0169  0.02  1
% 2  0.0069  0.57  0 <- smallest m where H0 accepted
% 3  0.0061  0.52  0
% 4  0.0057  0.41  0
% 5  0.0056  0.19  0
% 6  0.0052  0.17  0
% 7  0.0052  0.06  0

function [H, P, h] = bootmode (x, m, B, kernel, ncpus)

  % Store subfunctions in a stucture to make them available for parallel processes
  parsubfun = struct ('findCriticalBandwidth',@findCriticalBandwidth , ...
                      'kde',@kde);

  % Check if running in Octave (else assume Matlab)
  info = ver; 
  ISOCTAVE = any (ismember ({info.Name}, 'Octave'));
  if (nargin < 2)
    error (cat (2, 'bootmode usage: ''bootmode (X, M, varagin)'';', ...
                   ' a minimum of 2 input arguments are required'))
  end
  if (nargin < 3)
    B = '2000';
  end
  if (nargin < 4)
    kernel = 'Gaussian';
  end
  if (nargin < 5)
    ncpus = 0;    % Ignore parallel processing features
  elseif (~ isempty (ncpus))
    if (~ isa (ncpus, 'numeric'))
      error ('bootmode: NPROC must be numeric');
    end
    if (any (ncpus ~= abs (fix (ncpus))))
      error ('bootmode: NPROC must be a positive integer');
    end    
    if (numel (ncpus) > 1)
      error ('bootmode: NPROC must be a scalar value');
    end
  end
  if (ISOCTAVE)
    ncpus = min(ncpus, nproc);
  else
    ncpus = min(ncpus, feature('numcores'));
  end

  % If applicable, check we have parallel computing capabilities
  if (ncpus > 1)
    if (ISOCTAVE)
      pat = '^parallel';
      software = pkg ('list');
      names = cellfun (@(S) S.name, software, 'UniformOutput', false);
      status = cellfun (@(S) S.loaded, software, 'UniformOutput', false);
      index = find (~ cellfun (@isempty, regexpi(names,pat)));
      if (~ isempty(index))
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
        if (isempty (pool))
          if (ncpus > 1)
            % Start parallel pool with ncpus workers
            parpool (ncpus);
          else
            % Parallel pool is not running and ncpus is 1 so run
            % function evaluations in serial
            ncpus = 1;
          end
        else
          if (pool.NumWorkers ~= ncpus)
            % Check if number of workers matches ncpus and correct
            % it accordingly if not
            delete (pool);
            if (ncpus > 1)
              parpool (ncpus);
            end
          end
        end
      catch
        % MATLAB Parallel Computing Toolbox is not installed
        warning ('bootmode:parallel', cat (2, 'MATLAB Parallel Computing', ...
                 ' Toolbox is not installed or operational. Falling back', ...
                 ' to serial processing.'))
        ncpus = 1;
      end
    end
  else
    if ((ncpus > 1) && ~ PARALLEL)
      if (ISOCTAVE)
        % OCTAVE Parallel Computing Package is not installed or loaded
        warning ('bootmode:parallel', cat (2, 'OCTAVE Parallel Computing', ...
                ' Package is not installed and/or loaded. Falling back', ...
                ' to serial processing.'))
      else
        % MATLAB Parallel Computing Toolbox is not installed or loaded
        warning ('bootmode:parallel', cat (2, 'MATLAB Parallel Computing', ...
                 ' Toolbox is not installed and/or loaded. Falling back', ...
                 ' to serial processing.'))
      end
      ncpus = 0;
    end
  end

  % Vectorize the data
  x = x(:);
  n = numel(x);

  % Find critical bandwidth
  [criticalBandwidth] = parsubfun.findCriticalBandwidth (x, m, kernel);
  h = criticalBandwidth;

  % Random resampling with replacement from a smooth estimate of
  % the distribution
  idx = boot (n, B, false);
  Y = x(idx);
  xvar = var (x, 1); % calculate sample variance
  Ymean = ones (n, 1) * mean(Y);
  X = Ymean + (1 + h^2 / xvar)^(-0.5) * (Y - Ymean + (h * randn (n, B)));

  % Calculate bootstrap statistic of the bootstrap samples
  % The boostrap statistic is the number of modes
  if (ncpus > 1)
    % PARALLEL
    if (ISOCTAVE)
      % OCTAVE
      f = cell2mat (parcellfun (ncpus, ...
                                @(j) parsubfun.kde (X(:,j), h, kernel), ...
                                num2cell (1:B), 'UniformOutput', false));
    else
      % MATLAB
      f = zeros(200,B);
      parfor j = 1:B; f(:,j) = kde (X(:,j), h, kernel); end
    end
  else
    % SERIAL
    f = cell2mat (cellfun (@(j) parsubfun.kde (X(:,j), h, kernel), ...
                           num2cell (1:B), 'UniformOutput', false));
  end

  % Compute number of modes in the kernel density estimate of the
  % bootstrap samples
  mboot = sum (diff (sign (diff (f))) < 0);
  % Approximate achieved significance level (ASL) from bootstrapping
  % number of modes
  P = sum (mboot > m) / B;

  % Accept or reject null hypothesis
  if (P > 0.05)
    H = false;
  elseif (P < 0.05)
    H = true;
  end

end

%--------------------------------------------------------------------------

function [criticalBandwidth] = findCriticalBandwidth (x, mref, kernel)

  if (mref > numel (x))
    error ('the number of modes M cannot exceed the number of data points in X')
  end

  % Calculate starting value for bandwidth
  % The algorithm sets the initial bandwidth so that half
  % of the sorted, unique data points are well separated
  xsort = sort (x);
  xdiff = diff (xsort);
  h = median (xdiff (xdiff > 0)) / (2 * sqrt (2 * log (2)));

  m = inf;
  while m > mref
    % Increase h
    h = h * 1.01; % Increase query bandwidth by 1%
    y = kde (x, h, kernel);
    m = sum (diff (sign (diff (y))) < 0);
  end
  criticalBandwidth = h;

end

%--------------------------------------------------------------------------

function [f, t] = kde (x, h, kernel)

  % Vectorize the data
  x = x(:);

  % Default properties of t
  n = numel (x);
  tmin = min(x) - 3 * h; % define lower limit of kernel density estimate
  tmax = max(x) + 3 * h; % define upper limit of kernel density estimate

  % Kernel density estimator
  t = linspace (tmin,tmax,200)';
  f = zeros (200,1);
  for i = 1:n
    xi = ones (200,1) * x(i);
    u = (t - xi) / h;
    if (strcmpi (kernel, 'Epanechnikov'))
      % Epanechnikov (parabolic) kernel
      K = @(u) max(0, (3 / 4) * (1 - u.^2));
    elseif (strcmpi (kernel, 'Gaussian'))
      % Gaussian kernel
      K = @(u) 1 / sqrt (2 * pi) * exp (-0.5 * u.^2);
    end
    f = f + K(u);
  end
  f = f * 1 / (n * h);

end

%--------------------------------------------------------------------------

%!demo
%!
%! % Stamp data example used in reference [1] in bootstrap R package
%! x=[0.060;0.064;0.064;0.065;0.066;0.068;0.069;0.069;0.069;0.069;0.069; ...
%!    0.069;0.069;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070; ...
%!    0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070; ...
%!    0.070;0.070;0.070;0.070;0.070;0.070;0.071;0.071;0.071;0.071;0.071; ...
%!    0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071; ...
%!    0.071;0.071;0.071;0.071;0.072;0.072;0.072;0.072;0.072;0.072;0.072; ...
%!    0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072; ...
%!    0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072; ...
%!    0.072;0.072;0.072;0.073;0.073;0.073;0.073;0.073;0.073;0.073;0.073; ...
%!    0.073;0.073;0.073;0.074;0.074;0.074;0.074;0.074;0.074;0.074;0.074; ...
%!    0.074;0.074;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075; ...
%!    0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075; ...
%!    0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076; ...
%!    0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.077;0.077;0.077;0.077; ...
%!    0.077;0.077;0.077;0.077;0.077;0.077;0.077;0.078;0.078;0.078;0.078; ...
%!    0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078; ...
%!    0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.079;0.079;0.079; ...
%!    0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079; ...
%!    0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079; ...
%!    0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079; ...
%!    0.079;0.079;0.079;0.079;0.079;0.079;0.080;0.080;0.080;0.080;0.080; ...
%!    0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080; ...
%!    0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080; ...
%!    0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.081; ...
%!    0.081;0.081;0.081;0.081;0.081;0.081;0.081;0.081;0.081;0.081;0.081; ...
%!    0.081;0.081;0.081;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082; ...
%!    0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.083; ...
%!    0.083;0.083;0.083;0.083;0.083;0.083;0.084;0.084;0.084;0.085;0.085; ...
%!    0.086;0.086;0.087;0.088;0.088;0.089;0.089;0.089;0.089;0.089;0.089; ...
%!    0.089;0.089;0.089;0.089;0.090;0.090;0.090;0.090;0.090;0.090;0.090; ...
%!    0.090;0.090;0.091;0.091;0.091;0.092;0.092;0.092;0.092;0.092;0.093; ...
%!    0.093;0.093;0.093;0.093;0.093;0.094;0.094;0.094;0.095;0.095;0.096; ...
%!    0.096;0.096;0.097;0.097;0.097;0.097;0.097;0.097;0.097;0.098;0.098; ...
%!    0.098;0.098;0.098;0.099;0.099;0.099;0.099;0.099;0.100;0.100;0.100; ...
%!    0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100; ...
%!    0.100;0.101;0.101;0.101;0.101;0.101;0.101;0.101;0.101;0.101;0.102; ...
%!    0.102;0.102;0.102;0.102;0.102;0.102;0.102;0.103;0.103;0.103;0.103; ...
%!    0.103;0.103;0.103;0.104;0.104;0.105;0.105;0.105;0.105;0.105;0.106; ...
%!    0.106;0.106;0.106;0.107;0.107;0.107;0.108;0.108;0.108;0.108;0.108; ...
%!    0.108;0.108;0.109;0.109;0.109;0.109;0.109;0.109;0.109;0.110;0.110; ...
%!    0.110;0.110;0.110;0.110;0.110;0.110;0.110;0.110;0.110;0.111;0.111; ...
%!    0.111;0.111;0.112;0.112;0.112;0.112;0.112;0.114;0.114;0.114;0.115; ...
%!    0.115;0.115;0.117;0.119;0.119;0.119;0.119;0.120;0.120;0.120;0.121; ...
%!    0.122;0.122;0.123;0.123;0.125;0.125;0.128; 0.129;0.129;0.129;0.130;0.131];
%! 
%! [H1, P1, CRITVAL1] = bootmode (x,1,2000);
%!
%! % Repeat function call systematically increasing the number of modes (M) by 
%! % 1, until the null hypothesis is accepted (i.e. H0 = 0)
%!
%! [H2, P2, CRITVAL2] = bootmode (x,2,2000);
%! 
%! sprintf ('Summary of results:\n') 
%! sprintf (cat (2, 'H1 is %u with p = %.3g so reject the null hypothesis', ...
%!                  'that there is 1 mode\n'), H1, P1)
%! sprintf (cat (2, 'H2 is %u with p = %.3g so accept the null hypothesis', ...
%!                  ' that there are 2 modes\n'), H2, P2)
%! 
%! % Please be patient, these calculations take a while...

%!test
%!
%! % Stamp data example used in reference [1] in bootstrap R package
%! x=[0.060;0.064;0.064;0.065;0.066;0.068;0.069;0.069;0.069;0.069;0.069; ...
%!    0.069;0.069;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070; ...
%!    0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070; ...
%!    0.070;0.070;0.070;0.070;0.070;0.070;0.071;0.071;0.071;0.071;0.071; ...
%!    0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071; ...
%!    0.071;0.071;0.071;0.071;0.072;0.072;0.072;0.072;0.072;0.072;0.072; ...
%!    0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072; ...
%!    0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072; ...
%!    0.072;0.072;0.072;0.073;0.073;0.073;0.073;0.073;0.073;0.073;0.073; ...
%!    0.073;0.073;0.073;0.074;0.074;0.074;0.074;0.074;0.074;0.074;0.074; ...
%!    0.074;0.074;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075; ...
%!    0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075; ...
%!    0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076; ...
%!    0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.077;0.077;0.077;0.077; ...
%!    0.077;0.077;0.077;0.077;0.077;0.077;0.077;0.078;0.078;0.078;0.078; ...
%!    0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078; ...
%!    0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.079;0.079;0.079; ...
%!    0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079; ...
%!    0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079; ...
%!    0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079; ...
%!    0.079;0.079;0.079;0.079;0.079;0.079;0.080;0.080;0.080;0.080;0.080; ...
%!    0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080; ...
%!    0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080; ...
%!    0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.081; ...
%!    0.081;0.081;0.081;0.081;0.081;0.081;0.081;0.081;0.081;0.081;0.081; ...
%!    0.081;0.081;0.081;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082; ...
%!    0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.083; ...
%!    0.083;0.083;0.083;0.083;0.083;0.083;0.084;0.084;0.084;0.085;0.085; ...
%!    0.086;0.086;0.087;0.088;0.088;0.089;0.089;0.089;0.089;0.089;0.089; ...
%!    0.089;0.089;0.089;0.089;0.090;0.090;0.090;0.090;0.090;0.090;0.090; ...
%!    0.090;0.090;0.091;0.091;0.091;0.092;0.092;0.092;0.092;0.092;0.093; ...
%!    0.093;0.093;0.093;0.093;0.093;0.094;0.094;0.094;0.095;0.095;0.096; ...
%!    0.096;0.096;0.097;0.097;0.097;0.097;0.097;0.097;0.097;0.098;0.098; ...
%!    0.098;0.098;0.098;0.099;0.099;0.099;0.099;0.099;0.100;0.100;0.100; ...
%!    0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100; ...
%!    0.100;0.101;0.101;0.101;0.101;0.101;0.101;0.101;0.101;0.101;0.102; ...
%!    0.102;0.102;0.102;0.102;0.102;0.102;0.102;0.103;0.103;0.103;0.103; ...
%!    0.103;0.103;0.103;0.104;0.104;0.105;0.105;0.105;0.105;0.105;0.106; ...
%!    0.106;0.106;0.106;0.107;0.107;0.107;0.108;0.108;0.108;0.108;0.108; ...
%!    0.108;0.108;0.109;0.109;0.109;0.109;0.109;0.109;0.109;0.110;0.110; ...
%!    0.110;0.110;0.110;0.110;0.110;0.110;0.110;0.110;0.110;0.111;0.111; ...
%!    0.111;0.111;0.112;0.112;0.112;0.112;0.112;0.114;0.114;0.114;0.115; ...
%!    0.115;0.115;0.117;0.119;0.119;0.119;0.119;0.120;0.120;0.120;0.121; ...
%!    0.122;0.122;0.123;0.123;0.125;0.125;0.128; 0.129;0.129;0.129;0.130;0.131];
%! 
%! [H, P, CRITVAL] = bootmode (x,1,2000,'Gaussian');
%! assert (H, true);
%! [H, P, CRITVAL] = bootmode (x,2,2000,'Gaussian');
%! assert (H, false);
