% Computes credible interval(s) directly from a vector (or row-major matrix) of
% the posterior(s) obtained by bayesian bootstrap.
%
% -- Function File: CI = credint (BOOTSTAT)
% -- Function File: CI = credint (BOOTSTAT, PROB)
%
%     'CI = credint (BOOTSTAT)' computes 95% credible intervals directly from
%     the vector, or rows* of the matrix in BOOTSTAT, where BOOTSTAT contains
%     posterior (or Bayesian bootstrap) statistics, such as those generated
%     using the `bootbayes` function (or the `bootlm` function with the method
%     set to 'bayesian'). The credible intervals are shortest probability
%     intervals (SPI), which represent a more computationally stable version
%     of the highest posterior density interval [1,2].
%
%        * The matrix should have dimensions P * NBOOT, where P corresponds to
%          the number of parameter estimates and NBOOT corresponds to the number
%          of posterior (or Bayesian bootstrap) samples.
%
%     'CI = credint (BOOTSTAT, PROB)' returns credible intervals, where PROB is
%     numeric and sets the lower and upper bounds of the credible interval(s).
%     The value(s) of PROB must be between 0 and 1. PROB can either be:
%       <> scalar: To set the central mass of shortest probability intervals
%                  to 100*PROB%
%       <> vector: A pair of probabilities defining the lower and upper
%                  percentiles of the credible interval(s) as 100*(PROB(1))%
%                  and 100*(PROB(2))% respectively.
%          The default value of PROB is the scalar: 0.95, for a 95% shortest 
%          posterior credible interval.
%
%  Bibliography:
%  [1] Liu, Gelman & Zheng (2015). Simulation-efficient shortest probability
%        intervals. Statistics and Computing, 25(4), 809–819. 
%  [2] Gelman (2020) Shortest Posterior Intervals.
%        https://discourse.mc-stan.org/t/shortest-posterior-intervals/16281/16
%
%  credint (version 2023.09.03)
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

function CI = credint (Y, PROB)

  % Check input and output arguments
  if (nargin < 2)
    PROB = 0.95;
  end
  if (nargin > 2)
    error ('credint: Too many input arguments.')
  end
  if (nargout > 1)
    error ('credint: Too many output arguments.')
  end

  % Evaluate the dimensions of y
  sz = size (Y);
  if all (sz == 1)
    error ('credint: Y must be either a vector or a P * NBOOT matrix')
  end
  if (sz(2) == 1)
    nboot = sz(1);
    p = 1;
    Y = Y.';
  else
    p = sz(1);
    nboot = sz(2);
    if (p > nboot)
      warning ('credint: The dimensions of the matrix in Y should be P * NBOOT')
    end
  end

  % Evaluate PROB input argument
  if ( (nargin < 2) || (isempty (PROB)) )
    PROB = 0.95;
    nprob = 1;
  else
    nprob = numel (PROB);
    if (~ isa (PROB, 'numeric') || (nprob > 2))
      error ('credint: PROB must be a scalar or a vector of length 2')
    end
    if (size (PROB, 1) > 1)
      PROB = PROB.';
    end
    if (any ((PROB < 0) | (PROB > 1)))
      error ('credint: Value(s) in PROB must be between 0 and 1')
    end
    if (nprob > 1)
      % PROB is a pair of probabilities
      % Make sure probabilities are in the correct order
      if (PROB(1) > PROB(2) )
        error (cat (2, 'credint: The pair of probabilities must be in', ...
                       ' ascending numeric order'))
      end
    end
  end

  % Compute credible intervals
  % https://discourse.mc-stan.org/t/shortest-posterior-intervals/16281/16
  % This solution avoids fencepost errors
  CI = nan (p, 2);
  Y = sort (Y, 2);
  gap = round (PROB * (nboot + 1));
  for j = 1:p
    if (nprob > 1)
      % Percentile intervals
      if (~ isnan (PROB))
        CI(j, :) = Y(j, cat (2, max (1, gap(1)), min (nboot, gap(2))));
      end
      CI(:,isnan(PROB)) = NaN;
      if (gap(1) == 0)
        CI(:, 1) = -inf;
      end
      if (gap(2) > nboot)
        CI(:, 2) = +inf;
      end
    else
      % Shortest probability interval
      % This implementation ensures that if there are multiple minima, the
      % the shortest probability interval closest to the central interval is
      % chosen
      width = Y(j, (gap + 1):nboot) - Y(j, 1:(nboot - gap));
      index = find (width == min (width))';
      if (isempty (index))
        CI(j, :) = NaN;
      else
        best_index = index (dsearchn (index, 0.5 * (1 - PROB) * (nboot + 1)));
        if (~ isnan (PROB))
          CI(j, :) = Y(j, [best_index, best_index + gap]);
        end
      end
    end
  end

end

%!demo
%!
%! % Input univariate dataset
%! y = [5.18 2.71 2.69 6.09 4.23 10.7 3.71 13.13 19.28 6.61].';
%!
%! % 95% credible interval for the mean 
%! [stats, bootstat] = bootbayes (y);
%! CI = credint (bootstat,0.95)           % 95% shortest probability interval
%! CI = credint (bootstat,[0.025,0.975])  % 95% equal-tailed interval
%!
%! % Please be patient, the calculations will be completed soon...

%!test
%!
%! % Input univariate dataset
%! y = [5.18 2.71 2.69 6.09 4.23 10.7 3.71 13.13 19.28 6.61].';
%!
%! % 95% credible interval for the mean 
%! [stats, bootstat] = bootbayes (y);
%!
%! % 95% credible interval for the mean 
%! CI = credint (bootstat,0.95);          % 95% shortest probability interval
%! CI = credint (bootstat,[0.025,0.975]); % 95% equal-tailed interval
