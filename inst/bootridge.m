% Empirical Bayes penalized regression for univariate or multivariate outcomes, 
% with shrinkage tuned to minimize prediction error computed by .632 bootstrap.
%
% -- Function File: bootridge (Y, X)
% -- Function File: bootridge (Y, X, CATEGOR)
% -- Function File: bootridge (Y, X, CATEGOR, NBOOT)
% -- Function File: bootridge (Y, X, CATEGOR, NBOOT, ALPHA)
% -- Function File: bootridge (Y, X, CATEGOR, NBOOT, ALPHA, L)
% -- Function File: bootridge (Y, X, CATEGOR, NBOOT, ALPHA, L, DEFF)
% -- Function File: bootridge (Y, X, CATEGOR, NBOOT, ALPHA, L, DEFF, SEED)
% -- Function File: bootridge (Y, X, CATEGOR, NBOOT, ALPHA, L, DEFF, SEED, TOL)
% -- Function File: S = bootridge (Y, X, ...)
% -- Function File: [S, YHAT] = bootridge (Y, X, ...)
% -- Function File: [S, YHAT, P] = bootridge (Y, X, ...)
%
%      'bootridge (Y, X)' fits an empirical Bayes ridge regression model using
%      a linear Normal (Gaussian) likelihood with an empirical Bayes normal
%      ridge prior on the regression coefficients. The ridge tuning constant
%      (lambda) is optimized via .632 bootstrap-based machine learning (ML) to
%      minimize out-of-bag prediction error [1, 2]. Y is an m-by-q matrix of
%      outcomes and X is an m-by-n design matrix whose first column must
%      correspond to an intercept term. If an intercept term (a column of ones)
%      is not found in the first column of X, one is added automatically. If any
%      rows of X or Y contain missing values (NaN) or infinite values (+/- Inf),
%      the corresponding observations are omitted before fitting.
%
%      For each outcome, the function prints posterior summaries for regression
%      coefficients or linear estimates, including posterior means, equal-tailed
%      credible intervals, Bayes factors (lnBF10), and the marginal prior used
%      for inference. When multiple outcomes are fitted (q > 1), the function
%      additionally prints posterior summaries for the residual correlations
%      between outcomes, reported as unique (lower-triangular) outcome pairs.
%      For each correlation, the printed output includes the estimated
%      correlation and its credible interval.
%
%      Interpretation note (empirical Bayes):
%        Bayes factors reported by 'bootridge' are empirical‑Bayes approximations
%        based on a data‑tuned ridge prior. They are best viewed as model‑
%        comparison diagnostics (evidence on a predictive, information‑theoretic
%        scale) rather than literal posterior odds under a fully specified prior
%        [3–5]. The log scale (lnBF10) is numerically stable and recommended
%        for interpretation; BF10 may be shown as 0 or Inf when beyond machine
%        range, while lnBF10 remains finite. These Bayesian statistics converge 
%        to standard conjugate Bayesian evidence as the effective residual 
%        degrees of freedom (df_t) increase.
%
%      For convenience, the statistics-resampling package also provides the
%      function `bootlm`, which offers a user-friendly but feature-rich interface
%      for fitting univariate linear models with continuous and categorical
%      predictors. The design matrix X and hypothesis matrix L returned in the
%      MAT-file produced by `bootlm` can be supplied directly to `bootridge`.
%      The outputs of `bootlm` also provide a consistent definition of the model
%      coefficients, thereby facilitating interpretation of parameter estimates,
%      contrasts, and posterior summaries. The design matrix X and hypothesis
%      matrix L can also be obtained the same way with one of the outcomes of a
%      multivariate data set, then fit to all the outcomes using bootridge.
%
%      'bootridge (Y, X, CATEGOR)' specifies the predictor columns that
%      correspond to categorical variables. CATEGOR must be a scalar or vector
%      of integer column indices referring to columns of X (excluding the
%      intercept). Alternatively, if all predictor terms are categorical, set
%      CATEGOR to 'all' or '*'. CATEGOR does NOT create or modify dummy or
%      contrast coding; users are responsible for supplying an appropriately
%      coded design matrix X. The indices in CATEGOR are used to identify
%      predictors that represent categorical variables, even when X is already
%      coded, so that variance-based penalty scaling is not applied to these
%      terms.
%
%      For categorical predictors in ridge regression, use meaningful centered
%      and preferably orthogonal (e.g. Helmert or polynomial) contrasts whenever
%      possible, since shrinkage occurs column-wise in the coefficient basis.
%      Orthogonality leads to more stable shrinkage and tuning of the ridge
%      parameter. Although the prior is not rotationally invariant, Bayes
%      factors for linear contrasts defined via a hypothesis matrix (L) are
%      typically more stable when the contrasts defining the coefficients are
%      orthogonal.
%
%      'bootridge (Y, X, CATEGOR, NBOOT)' sets the number of bootstrap samples
%      used to estimate the .632 bootstrap prediction error. The bootstrap* has
%      first order balance to improve the efficiency for variance estimation,
%      and utilizes bootknife (leave-one-out) resampling to guarantee
%      observations in the out-of-bag samples. The default value of NBOOT is
%      100, but more resamples are recommended to reduce monte carlo error.
%
%      The bootstrap tuning of the ridge parameter relies on resampling
%      functionality provided by the statistics-resampling package. In
%      particular, `bootridge` depends on the functions `bootstrp` and `boot` to
%      perform balanced bootstrap and bootknife (leave-one-out) resampling and
%      generate out-of-bag samples. These functions are required for estimation
%      of the .632 bootstrap prediction error used to select the ridge tuning
%      constant.
%
%      'bootridge (Y, X, CATEGOR, NBOOT, ALPHA)' sets the central mass of equal-
%      tailed credibility intervals (CI) to (1 - ALPHA) with probability mass
%      ALPHA/2 in each tail, and sets the threshold for the adjusted stability
%      selection (SS) probabilities of the regression coefficients to (1 - ALPHA).
%      ALPHA must be a scalar value between 0 and 1. The default value of ALPHA
%      is 0.05 for 95% intervals.
%
%      'bootridge (Y, X, CATEGOR, NBOOT, ALPHA, L)' specifies a hypothesis
%      matrix L of size n-by-c defining c linear contrasts or model-based
%      estimates of the regression coefficients. In this case, posterior
%      summaries and credible intervals are reported for the linear estimates
%      rather than the model coefficients.
%
%      'bootridge (Y, X, CATEGOR, NBOOT, ALPHA, L, DEFF)' specifies a design
%      effect used to account for clustering or dependence. DEFF inflates the
%      posterior covariance and reduces the effective degrees of freedom (df_t) 
%      to ensure Bayes factors and intervals are calibrated for the effective 
%      sample size. For a mean, Kish's formula DEFF = 1+(g-1)*r (where g is 
%      cluster size) suggests an upper bound of g. However, for regression 
%      slopes, the realized DEFF depends on the predictor type: it can exceed 
%      g for between-cluster predictors or be less than 1 for within-cluster 
%      predictors. DEFF is best estimated as the ratio of clustered-to-i.i.d. 
%      sampling variances - please see DETAIL below. Default DEFF is 1.
%
%      'bootridge (Y, X, CATEGOR, NBOOT, ALPHA, L, DEFF, SEED)' initialises the
%      Mersenne Twister random number generator using an integer SEED value so
%      that bootstrap results are reproducible, which improves convergence.
%      Monte carlo error of the results can be assessed by repeating the
%      analysis multiple times, each time with a different random seed.
%
%      'bootridge (Y, X, CATEGOR, NBOOT, ALPHA, L, DEFF, SEED, TOL)' controls
%      the convergence tolerance for optimizing the ridge tuning constant lambda
%      on the log10 scale. Hyperparameter optimization terminates when the width
%      of the current bracket satisfies:
%
%          log10 (lambda_high) − log10 (lambda_low) < TOL.
%
%      Thus, TOL determines the relative (multiplicative) precision of lambda.
%      The default value TOL = 0.005 corresponds to approximately a 1% change in
%      lambda, which is typically well below the Monte Carlo noise of the .632
%      bootstrap estimate of prediction error.
%
%      * If sufficient parallel resources are available (three or more workers),
%        the optimization uses a parallel k‑section search; otherwise, a serial
%        golden‑section search is used. The tolerance TOL applies identically
%        in both cases. The benefit of parallel processing is most evident when
%        NBOOT is very large. In GNU Octave, the maximum number of workers used
%        can be set by the user before running bootridge, for example, for three
%        workers with the command:
%
%          setenv ('OMP_NUM_THREADS', '3')
%
%      'S = bootridge (Y, X, ...)' returns a structure containing posterior
%      summaries including posterior means, credibility intervals, Bayes factors,
%      prior summaries, the bootstrap-optimized ridge parameter, residual
%      covariance estimates, and additional diagnostic information.
%
%      The output S is a structure containing the following fields (listed in
%      order of appearance):
%
%        o coefficient
%            n-by-q matrix of posterior mean regression coefficients for each
%            outcome when no hypothesis matrix L is specified.
%
%        o estimate
%            c-by-q matrix of posterior mean linear estimates when a hypothesis
%            matrix L is specified. This field is returned instead of
%            'coefficient' when L is non-empty.
%
%        o CI_lower
%            Matrix of lower bounds of the (1 - ALPHA) credibility intervals
%            for coefficients or linear estimates. Dimensions match those of
%            'coefficient' or 'estimate'.
%
%        o CI_upper
%            Matrix of upper bounds of the (1 - ALPHA) credibility intervals
%            for coefficients or linear estimates. Dimensions match those of
%            'coefficient' or 'estimate'.
%
%        o BF10
%            Matrix of Bayes factors (BF10) for testing whether each regression
%            coefficient or linear estimate equals zero, computed using the
%            Savage–Dickey density ratio. Values may be reported as 0 or Inf
%            when outside floating‑point range; lnBF10 remains finite and is
%            the recommended evidential scale.
%
%        o lnBF10
%            Matrix of natural logarithms of the Bayes factors (BF10). Positive
%            values indicate evidence in favour of the alternative hypothesis,
%            whereas negative values indicate evidence in favour of the null.
%              lnBF10 < -1  is approx. BF10 < 0.3
%              lnBF10 > +1  is approx. BF10 > 3.0
%
%        o prior
%            Cell array describing the marginal inference-scale prior used for
%            each coefficient or estimate in Bayes factor computation.
%            Reported as 't (mu, sigma, df_t)' on the coefficient (or estimate)
%            scale; see CONDITIONAL VS MARGINAL PRIORS for details.
%
%        o lambda
%            Scalar ridge tuning constant selected by minimizing the .632
%            bootstrap estimate of prediction error (then scaled by DEFF).
%
%        o Sigma_Y_hat
%            Estimated residual covariance matrix of the outcomes, inflated by
%            the design effect DEFF when applicable. For a univariate outcome,
%            this reduces to the residual variance.
%
%        o df_lambda
%            Effective residual degrees of freedom under ridge regression,
%            defined as m minus the trace of the ridge hat matrix. Used for
%            residual variance estimation (scale); does NOT include DEFF.
%
%        o tau2_hat
%            Estimated prior covariance of the regression coefficients across
%            outcomes, proportional to Sigma_Y_hat and inversely proportional
%            to the ridge parameter lambda.
%
%        o Sigma_Beta
%            Cell array of posterior covariance matrices of the regression
%            coefficients. Each cell corresponds to one outcome and contains
%            the covariance matrix for that outcome.
%
%        o nboot
%            Number of bootstrap samples used to estimate the .632 bootstrap
%            prediction error.
%
%        o Deff
%            Design effect used to inflate the residual covariance and reduce
%            inferential degrees of freedom to account for clustering.
%
%        o tol
%            Numeric tolerance used in the golden-section search for optimizing
%            the ridge tuning constant.
%
%        o iter
%            Number of iterations performed by the golden-section search.
%
%        o pred_err
%            The minimized prediction error calculated using the optimal lambda.
%            Note that pred_err calculation is based on the outcome variables
%            (columns) in Y after internal standardization, and the predictors
%            X after internal centering.
%
%        o stability
%            The probabilities that the sign of the regression coefficients
%            remained consistent across max(nboot,1999) bootstrap resamples [6].
%            Raw probabilities are smoothed using a Jeffreys prior and, if
%            applicable, adjusted by the design effect (Deff). In the printed
%            summary, stability exceeding (1 - ALPHA / 2) is indicated by (+)
%            or (-) to denote the consistent direction of the effect.
%
%        o RTAB
%            Matrix summarizing residual correlations (strictly lower-
%            triangular pairs). The columns correspond to outcome J, outcome I, 
%            and the coefficient and credible intervals for their correlation.
%
%            Credible intervals for correlations are computed on Fisher’s z
%            [7] using a t‑based sampling distribution with effective degrees 
%            of freedom df_t, and then back‑transformed. See CONDITIONAL VS 
%            MARGINAL PRIORS and DETAIL below. Diagonal entries are undefined
%            and not included.
%
%      '[S, YHAT] = bootridge (Y, X, ...)' returns fitted values.
%
%      '[S, YHAT, P] = bootridge (Y, X, ...)' returns the predictor-wise penalty
%      weights used to normalize shrinkage across features of different scales.
%
%      DETAIL: The model implements an empirical Bayes ridge regression that
%      simultaneously addresses the problems of multicollinearity, multiple 
%      comparisons, and clustered dependence. The sections below provide
%      detail on the applications to which this model is well suited and the
%      principles of its operation.
%
%      REGULARIZATION AND MULTIPLE COMPARISONS: 
%      Unlike classical frequentist methods (e.g., Bonferroni) that penalize 
%      inference-stage decisions (p-values), `bootridge` penalizes the estimates 
%      themselves via shrinkage. By pooling information across all predictors to 
%      learn the global penalty (lambda), the model automatically adjusts its 
%      skepticism to the design's complexity. This provides a principled 
%      probabilistic alternative to family-wise error correction: noise-driven 
%      effects are shrunken toward zero, while stable effects survive the 
%      penalty. This "Partial Pooling" ensures that Bayes factors are 
%      appropriately conservative without the catastrophic loss of power 
%      associated with classical post-hoc adjustments [8, 9]. See later section
%      on STATISTICAL INFERENCE AND ERROR CONTROL.
%
%      PREDICTIVE OPTIMIZATION:
%      The ridge tuning constant (hyperparameter) is selected empirically by
%      minimizing the .632 bootstrap estimate of prediction error [1, 2]. This
%      aligns lambda with minimum estimated out‑of‑sample mean squared
%      prediction error, ensuring the model is optimized for generalizability
%      rather than mere in-sample fit [10–12]. This lambda in turn determines the
%      scale of the Normal ridge prior used to shrink slope coefficients toward
%      zero [13].
%
%      CONDITIONAL VS MARGINAL PRIORS:
%      The ridge penalty (lambda) corresponds to a Normal prior on the
%      regression coefficients CONDITIONAL on the residual variance:
%          Beta | sigma^2 ~ Normal(0, tau^2 * sigma^2),
%      where tau^2 is determined by lambda. This conditional Normal prior
%      fully defines the ridge objective function and is held fixed during
%      lambda optimisation (prediction-error minimisation).
%
%      For inference, however, uncertainty in the residual variance is
%      explicitly acknowledged. Integrating over variance uncertainty under
%      an empirical‑Bayes approximation induces a marginal Student’s t
%      distribution for coefficients and linear estimates, which is used
%      for credible intervals and Bayes factors.
%
%      PRIOR CALIBRATION & DATA INDEPENDENCE:
%      To prevent circularity in the prior selection, lambda is optimized 
%      solely by minimizing the .632 bootstrap out-of-bag (OOB) error. 
%      This ensures the prior precision is determined by the model's 
%      ability to predict "unseen" observations (data points not used 
%      for the coefficient estimation in a given bootstrap draw), 
%      thereby maintaining a principled separation between the data used 
%      for likelihood estimation and the data used for prior tuning.
%
%      STABILITY SELECTION:
%      The directional reproducibility of the sign of the regression coefficients
%      under resampling are quantified and reported as Stability Selection (SS).
%      It is possible for a shrunken coefficient to be highly stable in sign
%      despite having anecdotal Bayes Factors.
%
%      BAYES FACTORS:
%      For regression coefficients and linear estimates, Bayes factors are
%      computed using the Savage–Dickey density ratio evaluated on the
%      marginal inference scale. Prior and posterior densities are Student’s
%      t distributions with shared degrees of freedom (df_t), reflecting
%      uncertainty in the residual variance under an empirical‑Bayes
%      approximation [3–5].
%
%      For residual correlations between outcomes, credible intervals are 
%      computed on Fisher’s z [7] with effective degrees of freedom df_t and 
%      then back‑transformed to r.
%
%      SUMMARY OF PRIORS:
%      The model employs the following priors for empirical Bayes inference:
%
%        o Intercept: Improper flat/Uniform prior, U(-Inf, Inf).
%
%        o Slopes: Marginal Student’s t prior on the coefficient (or estimate)
%          scale, t(0, sigma_prior, df_t), with scale determined by the
%          bootstrap‑optimised ridge parameter (lambda) and design effect
%          DEFF.
%
%          In the limit (high df_t), the inferential framework converges to a 
%          Normal-Normal conjugate prior where the prior precision is 
%          determined by the optimized lambda. At lower df_t, the function 
%          provides more robust, t-marginalized inference to account for 
%          uncertainty in the error variance.
%
%        o Residual Variance: Implicit (working) Inverse-Gamma prior,
%          Inv-Gamma(df_t/2, Sigma_Y_hat), induced by variance estimation
%          and marginalization and used to generate the t-layer.
%
%        o Correlations: An improper flat prior is assumed on Fisher’s z
%          transform of the correlation coefficients. Under this prior, the
%          posterior for z is proportional to the t‑based sampling distribution
%          implied by the effective degrees of freedom df_t.
%
%      UNCERTAINTY AND CLUSTERING:
%      The design effect specified by DEFF is integrated throughout the model
%      consistent with its definition:
%              DEFF(parameter) =  Var_true(parameter) / Var_iid(parameter)
%      This guards against dependence between observations leading to anti-
%      conservative inference. This adjustment occurs at three levels:
%
%      1. Prior Learning: The ridge tuning constant (lambda) is selected by
%         minimizing predictive error on the i.i.d. bootstrap scale and then 
%         divided by DEFF. This "dilutes" the prior precision, ensuring the 
%             lambda_iid   = sigma^2 / tau^2_iid
%             tau^2_true   = DEFF * tau^2_iid
%             lambda_true  = sigma^2 / tau^2_true = lambda_iid / DEFF
%         where sigma^2 (a.k.a. Sigma_Y_hat) is residual variance (data space)
%         and tau^2 (a.k.a. tau2_hat) is the prior variance (parameter space).
%
%      2. Scale Estimation: Residual variance (Sigma_Y_hat) is estimated using
%         the ridge-adjusted degrees of freedom (df_lambda = m - trace(H_lambda))
%         and is then inflated by a factor of DEFF. This yields an "effective"
%         noise scale on the derived parameter statistics that accounts for
%         within-cluster correlation [14, 15] according to:
%             Var_true(beta_hat) = DEFF * Var_iid(beta_hat)
%
%      3. Inferential Shape: A marginal Student’s t layer is used for all
%         quantiles and Bayes factors to propagate uncertainty in the
%         residual variance and effective sample size. To prevent over-
%         certainty in small-cluster settings, the inferential degrees of
%         freedom are reduced: 
%             df_t = (m / DEFF) - trace (H_lambda), where m is size (Y, 1)
%         This ensures that both the scale (width) and the shape (tails) of the
%         posterior distributions are calibrated for the effective sample size.
%         The use of t‑based adjustments is akin to placing an Inverse-Gamma
%         prior (alpha = df_t / 2, beta = Sigma_Y_hat) on the residual variance
%         and is in line with classical variance component approximations (e.g.,
%         Satterthwaite/Kenward–Roger) and ridge inference recommendations
%         [16–18].
%
%      4. Stability Selection: The sign-consistency probabilities (denoted as
%         stability) under bootstrap resampling are adjusted for the design
%         effect via a Probit-link transformation: 
%            Phi ( Phi^-1(stability) / sqrt (Deff) )
%         Where Phi and Phi^-1 are the cumulative standard normal distribution
%         function and its inverse respectively. This adjustment ensures that
%         the reported stability reflects the effective sample size rather than
%         the raw number of observations, preventing over-certainty in the
%         presence of clustered or dependent data.
%
%      ESTIMATING THE DESIGN EFFECT:
%      While DEFF = 1 + (g - 1) * r provides a useful analytical upper bound 
%      based on cluster size (g) and intraclass correlation (r), the realized 
%      impact of dependence on regression slopes often varies by predictor type. 
%      For complex designs, DEFF is best estimated as the mean ratio of the 
%      parameter variances—obtained from the variances of the bootstrap 
%      distributions under a cluster-robust estimator (e.g., wild cluster 
%      bootstrap via `bootwild` or cluster-based bayesian bootstrap via 
%      `bootbayes`) relative to an i.i.d. assumption. Supplying this 
%      "Effective DEFF" allows `bootridge` to provide analytical Bayesian 
%      inference that approximates the results of a full hierarchical or 
%      resampled model [14, 15].
%
%      DIAGNOSTIC ASSESSMENT:
%      Users should utilize `bootlm` for formal diagnostic plots (Normal 
%      Q-Q, Spread-Location, Cook’s Distance). These tools identify 
%      influential observations that may require inspection before or 
%      after ridge fitting.
%
%      SUITABILITY: 
%      This function is designed for models with continuous outcomes and 
%      assumes a linear Normal (Gaussian) likelihood. It is not suitable for 
%      binary, count, or categorical outcomes. However, binary and categorical 
%      predictors are supported. 
%
%      INTERNAL SCALING AND STANDARDIZATION: 
%      All scaling and regularization procedures for optimizing the ridge
%      parameter are handled internally to ensure numerical stability and
%      balanced, scale-invariant shrinkage. To ensure all outcomes contribute 
%      equally to the global regularization regardless of their units, the 
%      ridge parameter (lambda) is optimized using internally standardized 
%      outcomes. 
%
%      When refitting the model with the optimal ridge parameter, while 
%      predictors are maintained on their original scale, the ridge penalty 
%      matrix is automatically constructed with diagonal elements proportional 
%      to the column variances of X. This ensures that the shrinkage applied 
%      to coefficients is equivalent to that of standardized predictors, 
%      without requiring manual preprocessing (categorical terms are identified 
%      via CATEGOR and are exempt from this variance-based penalty scaling). 
%      Following optimization, the final model is refit to the outcomes on 
%      their original scale; consequently, all posterior summaries, 
%      credibility intervals, and prior standard deviations are reported 
%      directly on the original coefficient scale for ease of interpretation.
%
%      STATISTICAL INFERENCE AND ERROR CONTROL:
%      Inference is provided via three complementary metrics: Credibility
%      Intervals (CI), Bayes Factors (BF), and Stability Selection (SS)
%      probabilities. Conditioned on a bootstrap-optimized ridge penalty, these
%      statistics exhibit superior control over Type M (magnitude) and Type S
%      (sign) errors relative to unpenalized estimators. The inherent shrinkage
%      provides implicit False Discovery Rate (FDR) control for CIs and BFs by
%      suppressing noise-driven inflation, providing more conservative global
%      error control than unpenalized methods. Conversely, SS probabilities
%      prioritize statistical power in sparse or low signal-to-noise ratio (SNR)
%      settings; while SS maintains marginal False Positive Rate (FPR) control
%      near ALPHA, it lacks the intrinsic FDR protection afforded by shrinkage
%      when interpreting multiple simultaneous inferences. The reliability of
%      all metrics improves as the Signal-to-Noise Ratio (SNR) increases. 
%
%                           CI           BF           SS   
%           FDR-Controlled <----------------------------> FPR-Controlled
%        (High Stringency)                                (High Discovery) 
%
%
%      See also: `bootstrp`, `boot`, `bootlm`, `bootbayes` and `bootwild`.
%
%  Bibliography:
%  [1] Delaney, N. J. & Chatterjee, S. (1986) Use of the Bootstrap and Cross-
%      Validation in Ridge Regression. Journal of Business & Economic Statistics,
%      4(2):255–262. https://doi.org/10.1080/07350015.1986.10509520
%  [2] Efron, B. & Tibshirani, R. J. (1993) An Introduction to the Bootstrap.
%      New York, NY: Chapman & Hall, pp. 247–252.
%      https://doi.org/10.1201/9780429246593
%  [3] Dickey, J. M. & Lientz, B. P. (1970) The Weighted Likelihood Ratio,
%      Sharp Hypotheses about Chances, the Order of a Markov Chain. Ann. Math.
%      Statist., 41(1):214–226. (Savage–Dickey)
%      https://doi.org/10.1214/aoms/1177697203
%  [4] Morris, C. N. (1983) Parametric Empirical Bayes Inference: Theory and
%      Applications. JASA, 78(381):47–55. https://doi.org/10.2307/2287098
%  [5] Wagenmakers, E.-J., Lodewyckx, T., Kuriyal, H., & Grasman, R. (2010) 
%      Bayesian hypothesis testing for psychologists: A tutorial on the 
%      Savage–Dickey method. Cognitive Psychology, 60(3):158–189.
%      https://doi.org/10.1016/j.cogpsych.2009.12.001
%  [6] Meinshausen, N. & Buhlmann, P. (2010) Stability selection. J. R. Statist.
%      Soc. B. 72(4): 417-473. https://doi.org/10.1111/j.1467-9868.2010.00740.x
%  [7] Fisher, R. A. (1921) On the "Probable Error" of a Coefficient of
%      Correlation Deduced from a Small Sample. Metron, 1:3–32. (Fisher z)
%  [8] Gelman, A., Hill, J., & Yajima, M. (2012) Why we usually don't worry 
%      about multiple comparisons. J. Res. on Educ. Effectiveness, 5:189–211.
%      https://doi.org/10.1080/19345747.2011.618213
%  [9] Efron, B. (2010) Large-Scale Inference: Empirical Bayes Methods for 
%      Estimation, Testing, and Prediction. Cambridge University Press.
%      https://doi.org/10.1017/CBO9780511761362
% [10] Hastie, T., Tibshirani, R., & Friedman, J. (2009) The Elements of
%      Statistical Learning (2nd ed.). Springer.
%      https://doi.org/10.1007/978-0-387-84858-7
% [11] Ye, J. (1998) On Measuring and Correcting the Effects of Data Mining and
%      Model Selection. JASA, 93(441):120–131. (Generalized df)
%      https://doi.org/10.1080/01621459.1998.10474094
% [12] Akaike, H. (1973) Information Theory and an Extension of the Maximum
%      Likelihood Principle. In: 2nd Int. Symp. on Information Theory. (AIC/KL)
%      https://doi.org/10.1007/978-1-4612-0919-5_38
% [13] Hoerl, A. E. & Kennard, R. W. (1970) Ridge Regression: Biased Estimation
%      for Nonorthogonal Problems. Technometrics, 12(1):55–67.
%      https://doi.org/10.1080/00401706.1970.10488634
% [14] Neuhaus, J. M., & Segal, M. R. (1993). Design effects for binary 
%      regression models fitted to dependent data. Statistics in Medicine, 
%      12(13):1259–1268. https://doi.org/10.1002/sim.4780121309
% [15] Cameron, A. C., & Miller, D. L. (2015) A Practitioner's Guide to 
%      Cluster-Robust Inference. J. Hum. Resour., 50(2):317–372.
%      https://doi.org/10.3368/jhr.50.2.317
% [16] Satterthwaite, F. E. (1946) An Approximate Distribution of Estimates of
%      Variance Components. Biometrics Bulletin, 2(6):110–114.
%      https://doi.org/10.2307/3002019
% [17] Kenward, M. G. & Roger, J. H. (1997) Small Sample Inference for Fixed 
%      Effects from Restricted Maximum Likelihood. Biometrics, 53(3):983–997.
%      https://doi.org/10.2307/2533558
% [18] Vinod, H. D. (1987) Confidence Intervals for Ridge Regression Parameters.
%      In Time Series and Econometric Modelling, pp. 279–300.
%      https://doi.org/10.1007/978-94-009-4790-0_19
%
% bootridge (version 2026.02.18)
% Author: Andrew Charles Penn


function [S, Yhat, P_vec] = bootridge (Y, X, categor, nboot, alpha, L, ...
                                    deff, seed, tol)

  % Check the number of input arguments provided
  if (nargin < 2)
    error (cat (2, 'bootridge: At least 2 input arguments, Y and X, required.'));
  end
  
  % Check the number of input arguments provided
  if (nargin > 9)
    error (cat (2, 'bootridge: Too many input arguments.'));
  end

  % Check that X and Y have the same number of rows
  if (size (Y, 1) ~= size (X, 1))
    error ('bootridge: the number of rows in X and Y must be the same');
  end

  % Omit any rows containing NaN or +/-Inf
  ridx = any ((isnan (cat (2, X, Y))) | (isinf (cat (2, X, Y))), 2);
  Y(ridx, :) = [];
  X(ridx, :) = [];

  % Get dimensions of the data
  [m, n] = size (X);
  q = size (Y, 2);

  % Check that the first column is X are all equal to 1, if not create one
  if ( ~all (X(:, 1) == 1) )
    X = cat (2, ones (m, 1), X);
    n = n + 1;
  end

  % If categor is not provided, set it to empty
  if ( (nargin < 3) || isempty (categor) )
    categor = [];
  else
    if ( any (strcmpi(categor, {'all', '*'})) )
      categor = (2:n);
    end
    if ( (~ isnumeric (categor)) || (sum (size (categor) > 1) > 1) || ...
         (any (isnan (categor))) || (any (isinf (categor))) || ...
         (any (categor < 1)) || (~ all (categor == abs (fix (categor)))) )
      error (cat (2, 'bootridge: categor should be a vector of column', ...
                     ' numbers corresponding to categorical variables'));
    end
    if ( ~ all (categor > 1) )
      error ('bootridge: The intercept should not be included in categor.')
    end
    if ( any (categor > n) )
      error ('bootridge: Numbers in categor exceed the number of columns in X');
    end
  end

  % If nboot is not specified, set it to 100.
  if ( (nargin < 4) || isempty (nboot) )
    nboot = 100;
  else
    if ((nboot <= 0) || (nboot ~= fix (nboot)) || isinf (nboot) || ...
         isnan (nboot) || (numel (nboot) > 1))
      error ('bootridge: nboot must be a finite positive integer');
    end
  end

  % If alpha is not specified, set it to .05.
  if ( (nargin < 5)|| isempty (alpha) )
    alpha = .05;
  else
    if ( ~isnumeric (alpha) || any (alpha <= 0) || any (alpha >= 1) || ...
         (numel (alpha) > 1) )
      error ('bootridge: Value of alpha must be between 0 and 1');
    end
  end

  % If a hypothesis matrix (L) is not provided, set it to empty
  if ( (nargin < 6) || isempty (L) )
    L = [];
    c = 0;
  else
    if (~ isempty (L))
      if (size (L, 1) ~= n)
        error (cat (2, 'bootridge: If not empty, L must have the same', ...
                       ' number of rows as columns in X.'));
      end
      c = size (L, 2);
    else
      c = 0;
    end
  end

  % If DEFF not specified, set it to 1
  if ( (nargin < 7) || isempty (deff) )
    deff = 1;
  else
    if ( (deff <= 0) || isinf (deff) || isnan (deff) || (numel (deff) > 1) )
      error ('bootridge: DEFF must be a finite scalar value > 0');
    end
  end

  % If seed not specified, set it to 1
  if ( (nargin < 8) || isempty (seed) )
    seed = 1;
  else
    if ( isinf (seed) || isnan (seed) || (numel (seed) > 1) || ...
         seed ~= fix(seed))
      error ('bootridge: The seed must be a finite integer');
    end
  end

  % If tol is not specified, set it to .01.
  if ( (nargin < 9)|| isempty (tol) )
    tol = 0.005;  % ~1% change in lambda
  else
    if ( ~isnumeric (tol) || isinf (tol) || isnan (tol) || (numel (tol) > 1))
      error ('bootridge: The tolerance must be a number');
    end
    tol = abs (tol);
  end

  % Check the number of output arguments requested
  if (nargout > 3)
    error ('bootridge: Only 3 output arguments can be requested.');
  end

  % Check if running in Octave (else assume Matlab)
  info = ver; 
  isoctave = any (ismember ({info.Name}, 'Octave'));

  % Check if we have parallel processing capabilities
  ncpus = 1;     % Default is serial processing
  if (isoctave)
    %%% OCTAVE %%%
    software = pkg ('list');
    names = cellfun (@(S) S.name, software, 'UniformOutput', false);
    status = cellfun (@(S) S.loaded, software, 'UniformOutput', false);
    index = find (~ cellfun (@isempty, regexpi (names, '^parallel')));
    if ( (~ isempty (index)) && (logical (status{index})) )
      % Set ncpus manually through environmental variable, for example:
      %   setenv ('OMP_NUM_THREADS', '4')
      % It is optimal to set this to the number of physical cores, which will be
      % different to the number of logical cores when hyperthreading is enabled. 
      % This can improve performance by reducing parallel overheads.
      ncpus = max (ncpus, nproc ('overridable'));
    end
  else
    %%% MATLAB %%%
    try 
      pool = gcp ('nocreate');
      if (~ isempty (pool))
        ncpus = max (ncpus, pool.NumWorkers);
      end
    catch
      % Do nothing
    end
  end

  % Create the penalty matrix - ridge regression will shrink all but intercept.
  % The penalty weight of each predictor term equals its variance. This makes
  % ridge shrinkage equivalent to what you would get if predictors were
  % standardized, ensuring the ridge parameter (lambda) applies uniformly across
  % predictors regardless of scale.
  P_vec = cat (2, 0, var (X(:,2:end), 0, 1))';

  % Evaluate categor input argument.
  if (~ isempty (categor))   
    % Set P_vec(k) to 1 where the predictor term corresponds to a categorical
    % variable. Categorical variable coding is exempt from penalty scaling.
    P_vec(categor) = 1;
  end

  % Objective function for lambda using .632 bootstrap prediction error.
  % Standardizing outcomes (YS) ensures equal weight across multivariate
  % dimensions
  z_score = @(A) bsxfun (@rdivide, bsxfun (@minus, A, mean (A)), std (A, 0));
  XC = X; 
  XC(:, 2:end) = bsxfun (@minus, X(:, 2:end), sum (X(:, 2:end), 1) / m);
  YS = z_score (Y);
  if ( any (isnan (YS(:)) | isinf (YS(:))) )
    error (cat (2, 'bootridge: Standardization requires outcomes to have', ...
                   ' nonzero variance.'));
  end
  parsubfun = struct ('booterr632', @booterr632, 'lambda_eval', @lambda_eval);
  obj_func = @(lambda) parsubfun.booterr632 (YS, XC, lambda, P_vec, nboot, seed);

  % Search for the optimal lambda by .632 bootstrap prediction error
  try
    smax = svds (XC, 1);                 % returns the largest singular value
  catch
    s = svd (XC);
    smax = s(1);
  end
  % Set search floor above numerical noise to avoid singular Cholesky factors.
  amin = log10 (smax^2 * min (m, n) * eps);
  bmax = log10 (smax^2);
  if (ncpus < 3)
    % Golden-section search (serial).
    [lambda, iter] = gss (obj_func, amin, bmax, tol);
  else
    % k-section search (parallel).
    % Only more efficient if ncpus is 4 or more.
    parsubfun.obj_func = obj_func;
    [lambda, iter] = kss (parsubfun, amin, bmax, tol, ncpus, isoctave);
  end

  % Get the prediction error and stability selection at the optimal lambda
  % Use a minimum of 1999 bootstrap resamples for stability selection
  B = max (nboot, 1999);
  [pred_err, stability] = booterr632 (YS, XC, lambda, P_vec, B, seed);

  % Correct stability selection probabilities for the design effect
  stdnormcdf = @(x) 0.5 * (1 + erf (x / sqrt (2)));
  stdnorminv = @(p) sqrt (2) * erfinv (2 * p - 1);
  z_stability = stdnorminv (stability);
  stability = stdnormcdf (z_stability / sqrt (deff));

  % Heuristic correction to lambda (prior precision) for the design effect.
  % Empirical-Bayes ridge learns lambda as an inverted estimator-scale SNR:
  %    lambda_iid ≈ sigma^2 / tau2_iid
  % where sigma^2 (a.k.a. Sigma_Y_hat) is residual variance (data space)
  % and tau^2 (a.k.a. tau2_hat) is the prior variance (parameter space).
  %
  % Under clustering, the information density is reduced. To maintain 
  % a consistent prior scale, the prior variance must be inflated:
  %    tau2_true = DEFF * tau2_iid
  %
  % Hence the EB precision learned under an i.i.d. assumption is too large:
  %    lambda_true = sigma^2 / tau2_true = lambda_iid / DEFF
  %
  % Thus, our apparent prior precision (lambda) under i.i.d. must be scaled
  % down by a factor of DEFF to prevent over-regularization.
  lambda = lambda / deff;

  % Regression coefficient and the effective degrees of freedom for ridge
  % regression penalized using the optimized (and corrected) lambda
  A = X' * X + diag (lambda * P_vec);     % Regularized normal equation matrix
  [U, flag] = chol (A);                   % Upper Cholesky factor of symmetric A
  tol = sqrt (m / eps (class (X)));       % Set tolerance  
  if (~ flag); flag = (max (diag (U)) / min (diag (U)) > 1e+06); end;
  if (flag)
    % Robust solve with pseudoinverse
    Beta = pinv (A) * (X' * Y);                   % n x q coefficient matrix
    df_lambda = m  - trace (pinv (A) * (X' * X)); % equivalent to m - trace (H)
  else
    % Fast solve by Cholesky decomposition
    Beta = U \ (U' \ (X' * Y));                   % n x q coefficient matrix
    df_lambda = m  - trace (U \ (U' \ (X' * X))); % Equivalent to m - trace (H)
  end
  df_lambda = max (df_lambda, 1);

  % Calculate the global, rotation‑invariant prior contribution as a percentage.
  % This is a ridge-based % prior contribution and it is relevant to the prior 
  % on coefficients and contrasts. A different formula is used for the prior on
  % the correlations between outcomes.
  r = rank (X(:, 2:end));              % rotation-invariant effective dimension.
  prior_perc_ridge = 100 * (1 - (m - df_lambda - 1 ) / r);

  % Calculate the residuals for the ridge fit using the optimal lambda
  resid = Y - X * Beta;                               % resid is m x q
  
  % Residual (co)variance (q x q) scaled by the design effect (Deff) to
  % propagate to all subsequent calculations of the prior and posterior.
  Sigma_Y_hat = deff * (resid' * resid) / df_lambda;

  % Prior (co)variance (matrix for q > 1)
  % The prior scale is effectively inflated by a factor of Deff^2
  tau2_hat = Sigma_Y_hat / lambda;

  % Posterior covariance (diagonal block) for the coefficients within outcome j;
  % Sigma_Beta{j} = Sigma_Y_hat(j,j) * invA
  invA = A \ eye (n);
  Sigma_Beta = arrayfun (@(j) Sigma_Y_hat(j, j) * invA, (1 : q), ...
                        'UniformOutput', false);

  % Distribution functions.
  % Student's (t) distribution:
  % A t-distribution for the prior and posterior is a mathematical approximation
  % in this empirical Bayes framework to having placed an Inverse-Gamma prior on
  % the variance and integrating it out.
  if ((exist ('betaincinv', 'builtin')) || (exist ('betaincinv', 'file')))
    distinv = @(p, df) sign (p - 0.5) * ...
                sqrt ( df ./ betaincinv (2 * min (p, 1 - p), df / 2, 0.5) - df);
  else
    % Earlier versions of Matlab do not have betaincinv
    % Instead, use betainv from the Statistics and Machine Learning Toolbox
    try 
      distinv = @(p, df) sign (p - 0.5) * ...
                  sqrt ( df ./ betainv (2 * min (p, 1 - p), df / 2, 0.5) - df);
    catch
      % Use critical values from the Normal distribution if either betaincinv
      % or betainv are not available
      distinv = @(p, df) sqrt (2) * erfinv (2 * p - 1);
      warning ('bootridge:', ...
      'Could not create studinv function; intervals will use z critical value');
    end
  end
  distpdf = @(t, mu, v, df) (exp (gammaln ((df + 1) / 2) - ...
                             gammaln (df / 2)) ./ sqrt(df * pi * v)) .* ...
                             (1 + (t - mu).^2 ./ (df .* v)).^(-(df + 1) / 2);
  % Normal (z) distribution (df input argument is ignored):
  %distinv = @(p, df) sqrt (2) * erfinv (2 * p - 1);
  %distpdf = @(z, mu, v, df) exp (-0.5 * ((z - mu).^2) ./ v) ./ sqrt (2 * pi * v);
  
  % Set critical value for credibility intervals
  % Effective inferential degrees of freedom for marginal (variance‑integrated)
  % inference; this does NOT affect ridge optimisation, only uncertainty and
  % Bayes factors.
  df_t = max (1, m / deff - trace (A \ (X' * X)));   %  m / DEFF - trace (H)
  if (df_t == 1)
    fprintf (cat (2, 'Note: t-statistics evaluated with effective', ...
                     ' degrees of freedom clamped at 1 degree of freedom.\n'));
  end
  critval = distinv (1 - alpha / 2, df_t); % Student's t distribution
  %critval = stdnorminv (1 - alpha / 2);    % Use Normal z distribution

  % Calculation of credibility intervals
  if (c < 1)

    % Calculation of credibility intervals for model coefficients
    CI_lower = zeros (n, q);
    CI_upper = zeros (n, q);
    for j = 1:q
      % Calculation of posterior statistics for outcome j
      se_j = sqrt (diag (Sigma_Beta{j})); % Standard deviation of the posterior
      CI_lower(:,j) = Beta(:,j) - critval .* se_j;
      CI_upper(:,j) = Beta(:,j) + critval .* se_j;
    end

    % Calculations for reporting Bayes factors for regression coefficients.
    % Prior variance and standard deviation for each coefficient (rows) and
    % outcome (columns). Report mean and standard deviation of the normal 
    % distribution used for the prior.
    ridx = false (n, 1); ridx(1) = true;
    V0 = bsxfun (@rdivide, diag (tau2_hat)', P_vec); V0(ridx,:) = Inf; 
    prior = arrayfun (@(v) sprintf('t (0, %#.3g, %#.3g)', ...
                      sqrt (v), df_t), V0, 'UniformOutput', false);
    %prior = arrayfun (@(v) sprintf('N (0, %#.3g)', ...
    %                  sqrt (v)), V0, 'UniformOutput', false);
    prior(ridx,:) = {'U (-Inf, Inf)'};

    % Marginal posterior variances for each coefficient-outcome pair
    V1 = diag (invA) * diag (Sigma_Y_hat)';

    % Marginal posterior density at 0 for each coefficient/outcome
    % Note that the third input argument is variance, not standard deviation.
    pH1 = distpdf (0, Beta, V1, df_t);

  else

    % Calculation of credibility intervals for model-based estimates
    mu = L' * Beta;  % c x q matrix
    CI_lower = zeros (c, q);
    CI_upper = zeros (c, q);
    for j = 1:q
      % Posterior variance for linear estimates for outcome j
      se_j = sqrt (diag (L' * Sigma_Beta{j} * L));
      CI_lower(:,j) = mu(:,j) - critval .* se_j;
      CI_upper(:,j) = mu(:,j) + critval .* se_j;
    end

    % Calculations for reporting Bayes factors for linear estimates.
    % Prior variance and standard deviation for each estimate (rows) and
    % outcome (columns). Report mean and standard deviation of the normal 
    % distribution used for the prior.
    ridx = ( abs (L(1,:)') > eps );
    P_inv_vec = zeros (n, 1);
    P_inv_vec(2:end) = 1 ./ P_vec(2:end);
    P_L = sum (bsxfun (@times, L.^2, P_inv_vec), 1)'; % diag (L' * pinv (P) * L)
    V0 = bsxfun (@times, diag (tau2_hat)', P_L); V0(ridx, :) = Inf;
    prior = arrayfun (@(v) sprintf('t (0, %#.3g, %#.3g)', ...
                      sqrt (v), df_t), V0, 'UniformOutput', false);
    %prior = arrayfun (@(v) sprintf('N (0, %#.3g)', ...
    %                  sqrt (v)), V0, 'UniformOutput', false);
    prior(ridx, :) = repmat ({'U (-Inf, Inf)'}, nnz (ridx), q);

    % Marginal posterior variances for each linear estimate/outcome
    invA_L = L' * invA * L;
    V1 = diag (invA_L) * diag (Sigma_Y_hat)';              % c x q

    % Marginal posterior density at 0 for each linear estimate/outcome
    % Note that the third input argument is variance, not standard deviation.
    pH1 = distpdf (0, mu, V1, df_t);

  end

  % Marginal prior density at 0 for each estimate or coefficient per outcome
  %pH0 = (2 * pi * V0).^(-0.5);      % Applies only to a Normal distribution
  pH0 = distpdf (0, 0, V0, df_t);

  % Bayes factor (Savage–Dickey ratio): relative plausibility (density) of the
  % coefficient or estimate being exactly 0 under the prior compared to under
  % the posterior.
  % BF10 > 1: The data have made observing Beta = 0 LESS plausible.
  %           This is evidence in favour of the alternative hypothesis.
  % BF10 < 1: The data have made observing Beta = 0 MORE plausible.
  %           This is evidence in favour of the null hypothesis.
  BF10   = bsxfun (@rdivide, pH0, pH1); BF10(ridx,:) = NaN;
  lnBF10 = log (BF10);                 % ln(1) = 0, ln(0.3) ~= -1, ln(3) ~= +1

  % Credible intervals for correlations between outcomes
  if (q > 1)

    % Posterior correlation between outcomes
    d = sqrt (diag (Sigma_Y_hat));
    R = Sigma_Y_hat ./ (d * d');
    R = R(tril (true (size (R)), -1));

    % Fisher's z-transform with numerical guard
    Z = atanh (min (max (R, -1 + eps), 1 - eps));

    % Effective sample size for Fisher‑z variance implied by inferential df_t
    n_eff = df_t;

    % Standard error of Z
    SE_z = 1 / sqrt (max (n_eff, 3) - 2);

    % Credible intervals under a flat (improper) prior on Fisher’s z.
    % Since the posterior is proportional to the marginal-t likelihood, 
    % equal-tailed intervals are obtained from z_obs ± t_ν * SE_z.
    R_CI_lower = tanh (Z - critval * SE_z);
    R_CI_upper = tanh (Z + critval * SE_z);

    % Get indices of pairs of outcomes
    [I, J] = find (tril (true (q), -1));

    % Assemble table-like cell array for correlations
    RTAB = cat (2, J, I, R, R_CI_lower, R_CI_upper);

  end

  % Pack results
  if (c < 1); S.coefficient = Beta; else; S.estimate = mu; end;
  S.CI_lower = CI_lower;
  S.CI_upper = CI_upper;
  S.BF10 = BF10;
  S.lnBF10 = lnBF10;
  S.prior = prior;
  S.lambda = lambda;
  S.Sigma_Y_hat = Sigma_Y_hat;
  S.df_lambda = df_lambda;
  S.tau2_hat = tau2_hat;
  S.Sigma_Beta = Sigma_Beta;
  S.nboot = nboot;
  S.Deff = deff;
  S.tol = tol;
  S.iter = iter;
  S.pred_err = pred_err;
  S.stability = stability;
  if (q > 1); S.RTAB = RTAB; end
  if (nargout > 1)
    YHAT = X * Beta;
  end

  % Display summary
  if (nargout == 0)
    fprintf (cat (2, '\n Empirical Bayes Ridge Regression (.632 Bootstrap',...
                     ' Tuning) - Summary\n ******************************', ...
                     '*************************************************\n'));
    fprintf ('\n Number of outcomes (q): %d\n', q);
    fprintf ('\n Design effect (Deff): %.3g\n', deff);
    fprintf('\n Bootstrap resamples (nboot): %d\n', nboot);
    fprintf('\n Minimized .632 bootstrap prediction error: %.6g\n', pred_err);
    if (deff == 1)
      fprintf (cat (2, '\n Bootstrap optimized ridge tuning constant', ...
                       ' (lambda): %.6g\n'), lambda);
    else
      fprintf (cat (2, '\n Bootstrap optimized ridge tuning constant', ...
                       ' (lambda, Deff-adjusted): %.6g\n'), lambda);
    end
    fprintf (cat (2, '\n Effective residual degrees of freedom (df_lambda):', ...
                     ' %#.3g\n'), df_lambda);
    if (q > 1)
      % Just print the average or the range of residual variances (the diagonal)
      res_vars = diag (Sigma_Y_hat);
      if (deff == 1)
        fprintf ('\n Residual variances (range): [%.3g, %.3g]\n', ...
                 min(res_vars), max(res_vars));
      else
        fprintf (cat (2, '\n Residual variances (range, Deff-inflated): ', ... 
                         ' [%.3g, %.3g]\n'), min(res_vars), max(res_vars));
      end
    else
      if (deff == 1)
        fprintf ('\n Residual variance (sigma^2): %.3g\n', Sigma_Y_hat);
      else
        fprintf ('\n Residual variance (sigma^2, Deff-inflated): %.3g\n', ...
                 Sigma_Y_hat);
      end
    end
    if (deff == 1)
      fprintf ('\n Inferential degrees of freedom (df_t): %#.3g', df_t);
    else
      fprintf (cat (2, '\n Inferential degrees of freedom (df_t,', ...
                       ' Deff-adjusted): %#.3g'), df_t);
    end
    fprintf ('\n Used for credible intervals and Bayes factors below.\n')

    % Correlations between outcomes
    if (q > 1)
      fprintf (cat (2, '\n %.3g%% credible intervals for', ...
                       ' correlations between outcomes:\n'), ...
                       100 * (1 - alpha));
      fprintf (' (Prior on Fisher''s z is flat/improper)\n')
      fprintf (cat (2, '\n Outcome J     Outcome I     correlation', ...
                        '   CI_lower      CI_upper     \n'));
      for i = 1:q*(q-1)*0.5
        fprintf (' %-10d    %-10d    %#-+9.4g     %#-+9.4g     %#-+9.4g\n', ...
                 J(i), I(i), R(i), R_CI_lower(i), R_CI_upper(i));
      end
    end

    % Coefficients and linear estimates
    if (c < 1)
      fprintf (cat (2, '\n %.3g%% credible intervals and Bayes factors', ...
                       ' for regression coefficients.\n'), ...
                       100 * (1 - alpha));
      fprintf (cat (2, ' Global ridge prior contribution to posterior ', ... 
                       'precision: %#.2f %%\n'), prior_perc_ridge);
      fprintf (cat (2, ' Stability selection (SS): >%.3g%% for the (-) or ', ...
                       '(+) sign of the coefficient.\n'), 100 * (1 - alpha / 2));
      if (deff == 1)
        fprintf (' SS probabilities are reported in S.stability\n');
      else
        fprintf (' Deff-adjusted SS probabilities are reported in S.stability\n');
      end
      for j = 1:q
        fprintf (cat (2, '\n Outcome %d:\n coefficient   CI_lower      ', ...
                         'CI_upper      lnBF10    SS  prior\n'), j);
        for k = 1:n
          if (stability(k, j) > (1 - alpha / 2))
            if (Beta(k, j) < 0)
              ss = '(-)';
            elseif (Beta(k, j) > 0)
              ss = '(+)';
            end 
          else
            ss = '   ';
          end
          fprintf (cat (2, ' %#-+10.4g    %#-+10.4g    %#-+10.4g    ', ...
                           '%#-+9.3g %s %s\n'), ...
                  Beta(k, j), CI_lower(k, j), CI_upper(k, j), lnBF10(k, j), ...
                  ss, prior{k, j});
        end
      end
    else
      fprintf (cat (2, '\n %.3g%% credible intervals and Bayes factors', ...
                       ' for linear estimates.\n'), ...
                       100 * (1 - alpha));
      fprintf (cat (2, ' Global ridge prior contribution to posterior ', ... 
                       'precision: %#.2f %%\n'), prior_perc_ridge);
      for j = 1:q
        fprintf (cat (2, '\n Outcome %d:\n estimate      CI_lower      ', ...
                         'CI_upper      lnBF10        prior\n'), j);
        for k = 1:c
          fprintf (' %#-+10.4g    %#-+10.4g    %#-+10.4g    %#-+9.3g     %s\n', ...
                  mu(k, j), CI_lower(k, j), CI_upper(k, j), lnBF10(k, j), ...
                  prior{k, j});
        end
      end
    end
    fprintf('\n');

  end

end


%-------------------------------------------------------------------------------

%% FUNCTION FOR .632 BOOTSTRAP ESTIMATOR OF PREDICTION ERROR

function [PRED_ERR, STABILITY] = booterr632 (Y, X, lambda, P_vec, nboot, seed)

  % This function computes Efron & Tibshirani’s .632 bootstrap prediction error
  % for a multivariate linear ridge/Tikhonov model. Loss is the per-observation
  % squared Euclidean error:
  %       Q(y_i, yhat_i) = ||y_i - yhat_i||_2^2
  % Efron and Tibshirani (1993) An Introduction to the Bootstrap. New York, NY:
  %  Chapman & Hall. pg 247-252
  
  % Calculate dimensions
  [m, q] = size (Y);
  n = size (X, 2);

  % Generate balanced bootstrap indices
  BOOTSAM = boot (m, nboot, true, seed);

  % --- HYBRID STRATEGY SELECTION ---
  % If n >  m, the Primal matrix (n x n) is bigger than the Dual matrix (m x m)
  % If m >= n, the Primal matrix (n x n) is the same or smaller than the Dual.
  use_dual = (n > m);

  % --- PRE-COMPUTATION PHASE ---
  eps_X = eps (class (X));
  if (use_dual)
      % DUAL SETUP: Prepare for Woodbury Solve (m x m inversion)
      % Calculate inverse of P_vec, dropping unpenalized intercept (from K)
      % YS columns are standardized so intercept is redundant for the prediction
      % error objective anyway.
      P_vec(1) = Inf;  % Intercept gets "infinite" prior precision (unpenalized)
      P_inv_vec = 1 ./ P_vec; % Result: P_inv_vec(1) is exactly 0.0
      % Set the tolerance for ill-conditioned system matrix. 
      % The noise floor for each element in KCi (m x m) is goverened by n.
      tol = sqrt (n / eps_X);
  else
      % PRIMAL SETUP: Prepare for Standard Solve (n x n inversion)
      % Regularization matrix
      LP = diag (lambda * P_vec);
      % Set the tolerance for ill-conditioned system matrix.
      % The noise floor for each element in A (n x n) is goverened by m.
      tol = sqrt (m / eps_X);
  end

  % Apparent error in resamples (A_ERR)
  % Re-fit on full data using the selected efficient method. Note that Y
  % provided as input must already be standardized, and X provided must already
  % be centered.
  if (use_dual)
    XW = bsxfun (@times, X, sqrt (P_inv_vec)');
    K  = XW * XW';
    Kr = K; Kr(1:m+1:end) = Kr(1:m+1:end) + lambda;  % Regularized kernel
    [U, flag] = chol (Kr);              % Upper Cholesky factor of symmetric Kr
    if (~ flag); flag = (max (diag (U)) / min (diag (U)) > tol); end;
    if (flag)
      Alpha_obs = pinv (Kr) * Y;        % Fail-safe solve
    else
      Alpha_obs = U \ (U' \ Y);         % Fast solve
    end
    Beta_obs = bsxfun (@times, P_inv_vec, (X' * Alpha_obs));
  else
    A = X' * X + LP;                    % Regularized normal equation matrix
    [U, flag] = chol (A);               % Upper Cholesky factor of symmetric A
    if (~ flag); flag = (max (diag (U)) / min (diag (U)) > tol); end;
    if (flag)
      Beta_obs = pinv (A) * (X' * Y);   % Fail-safe solve
    else
      Beta_obs = U \ (U' \ (X' * Y));   % Fast solve
    end
  end
  RESI = Y - X * Beta_obs;
  A_ERR = sum (RESI(:).^2) / m;

  % --- BOOTSTRAP LOOP ---
  SSE_OOB  = 0; 
  N_OOB    = 0;
  if (nargout > 1)
    tau = sqrt (eps_X);
    Sign_obs = sign (Beta_obs);
    STABILITY = zeros (n, q);
  end
  for b = 1:nboot
 
    % Get resampled indices for resample b
    i = BOOTSAM(:, b);

    % Find the indices of OOB observations for this bootstrap resample
    o = true (m, 1);
    o(i) = false;

    % Check for missing predictors in training set and remove out-of-bag
    % samples that have those predictors
    missing = ~ any (X(i, :), 1);
    if any (missing)
       o(any (X(:,missing), 2)) = false;
    end

    % Skip to next bootstrap sample if there are no out-of-bag observations
    if (~ any(o)); continue; end

    % Algorithm for calculation of Beta is dependent on the data dimensions.
    % The ridge parameter, lambda, helps to prevent the system matrix from
    % becoming singular, so we can use very fast Cholesky decomposition in dual
    % or primal space depending on the dimensions of X.
    if (use_dual)
        % DUAL (WOODBURY) SOLVE (Fast for n > m)
        % Solve the m x m system without the intercept
        % Build the "Local" Kernel (KCi) from centered data
        % This ensures rank is exactly the same as the Primal XCi' * XCi
        Xi = X(i, :);
        Yi = Y(i, :);
        mx = sum (Xi, 1) / m;
        my = sum (Yi, 1) / m;
        XCi = bsxfun (@minus, Xi, mx);
        YCi = bsxfun (@minus, Yi, my);
        XWi = bsxfun (@times, XCi, sqrt(P_inv_vec)'); 
        KCi = XWi * XWi'; 
        KCi(1:m+1:end) = KCi(1:m+1:end) + lambda;
        [U, flag] = chol (KCi);         % Upper Cholesky factor of symmetric KCi
        if (~ flag); flag = (max (diag (U)) / min (diag (U)) > tol); end;
        if (flag)
          Alpha = pinv (KCi) * YCi;     % Fail-safe solve
        else
          Alpha = U \ (U' \ YCi);       % Fast solve by Cholesky decomposition
        end
        XCo = bsxfun (@minus, X(o, :), mx);
        XWo = bsxfun (@times, XCo, sqrt (P_inv_vec)');
        PRED_OOB = bsxfun (@plus, (XWo * XWi') * Alpha, my);
        if (nargout > 1)
          Beta = bsxfun (@times, P_inv_vec, (XCi' * Alpha));
          selected = (sign (Beta) == Sign_obs) & (abs (Beta) > tau);
          selected (1, :) = false; % Ignore the intercept
          STABILITY  = STABILITY + selected;
        end
    else
        % PRIMAL (STANDARD) SOLVE (Fast for m >= n)
        % Primal solve: (n x n)
        A = (X(i, :)' * X(i, :) + LP);  % Regularized normal equation matrix
        [U, flag] = chol (A);           % Upper Cholesky factor of symmetric A
        if (~ flag); flag = (max (diag (U)) / min (diag (U)) > tol); end;
        if (flag)
          Beta = pinv (A) * (X(i, :)' * Y(i, :)); % Fail-safe solve
        else
          Beta = U \ (U' \ (X(i, :)' * Y(i, :))); % Fast solve
        end
        % Predict the values of the OOB observations from the coefficients for
        % for each of the q outcomes
        PRED_OOB = X(o, :) * Beta;
        if (nargout > 1)
          selected = (sign (Beta) == Sign_obs) & (abs (Beta) > tau);
          selected (1, :) = false; % Ignore the intercept
          STABILITY  = STABILITY + selected;
        end
    end

    % Calculate the residuals of the OOB predictions for each of the outcomes
    RESI_OOB = Y(o, :) - PRED_OOB;

    % Calculate and accumulate the per-observation squared Euclidean residuals
    SSE_OOB = SSE_OOB + sum (RESI_OOB(:).^2); % Eq. to sum(sum(RESI_OOB.^2,2))

    % Calculate and accumulate number of OOB observations
    N_OOB  = N_OOB + sum (o) ;

  end

  % Calculate pooled OOB error estimate
  S_ERR = SSE_OOB / N_OOB;

  % Optimism in apparent error (OPTIM)
  OPTIM = .632 * (S_ERR - A_ERR);

  % The bootstrap .632 estimator of prediction error
  PRED_ERR = A_ERR + OPTIM;

  % Calculate stability selection
  if (nargout > 1)
    % Convert counts to proportions, with Jeffrey's smoothing.
    STABILITY = (STABILITY + 0.5) / (nboot + 1.0);
    STABILITY(1, :) = NaN;  % Set stability selection to NaN for the intercepts
  end

end

%-------------------------------------------------------------------------------

% FUNCTION FOR GOLDEN-SECTION SEARCH MINIMIZATION OF AN OBJECTIVE FUNCTION

function [lambda, iter] = gss (f, a, b, tol)

  % After initialization, golden-section search only requires a single function
  % evaluation per iteration.
  % Algorithm based on https://en.wikipedia.org/wiki/Golden-section_search

  % Initialization
  invphi = (sqrt (5) - 1) / 2;
  iter = 0;

  % Start iterative optimization
  while ((b - a) > tol)
    if (iter < 1)
      c = b - (b - a) * invphi;
      d = a + (b - a) * invphi;
      fc = f(10^c);
      fd = f(10^d);
    end
    if (fc < fd)
      % Minimum lies in [a, d]; reuse fc
      b = d; d = c; fd = fc;
      c = b - (b - a) * invphi;
      fc = f(10^c);
    else
      % Minimum lies in [c, b]; reuse fd
      a = c; c = d; fc = fd;
      d = a + (b - a) * invphi;
      fd = f(10^d);
    end
    iter = iter + 1;
  end

  % Calculate return value
  lambda = 10^((b + a)/2);

end

%--------------------------------------------------------------------------

% FUNCTION FOR K-SECTION SEARCH MINIMIZATION OF AN OBJECTIVE FUNCTION

function [lambda, iter] = kss (parsubfun, a, b, tol, k, isoctave)

  % Parallel k-section search. 

  % Make k odd
  k = k - 1 + mod (k, 2);

  % Initialization
  iter = 0;
  c = 0;
  fxc = Inf;

  % Start iterative optimization
  while ((b - a) > tol)

    % Create vector of lambda values x
    p  = arrayfun (@(i) a + i / (k + 1) * (b - a), 1:k);
    x = 10.^p;

    % Perform function evaluations (skipping the middle value when iter > 0)
    if (isoctave)
      fx = pararrayfun (k, @(i) parsubfun.lambda_eval (x, i, c, fxc, ...
                                                       parsubfun), 1:k);
    else
      fx = nan (1, k);
      parfor i = 1:k
        if (i == c)
          fx(i) = fxc;
        else
          fx(i) = parsubfun.obj_func(x(i));
        end
      end
    end

    % Check for "Flatness" of the objective function across the search point
    if (max (fx) - min (fx) < eps)
      break
    end

    % Index of element with the minimum value of f(x)
    [fxc, m] = min (fx);

    % Boundary updates
    if (m == 1)
      b = p(2);
    elseif (m == k)
      a = p(k - 1);
    else
      a = p(m - 1); b = p(m + 1);
    end
    c = (k + 1) / 2;     % Index at the centre of the search
    iter = iter + 1;

  end

  % Calculate return value
  lambda = 10^((b + a)/2);

end

%-------------------------------------------------------------------------------

function fx = lambda_eval (x, i, c, fxc, parsubfun)

    % Helper function for kss for conditional evaluate of a vactor (x) of lambda
    % If i equals c use fxc, otherwise evaluate f(x(i))
    % where f is parsubfun.obj_func()
    if (i == c)
      fx = fxc;
    else
      fx = parsubfun.obj_func(x(i));
    end

end

%-------------------------------------------------------------------------------

%!demo
%!
%! % Simple linear regression. The data represents salaries of employees and
%! % their years of experience, modified from Allena Venkata. The salaries are
%! % in units of 1000 dollars per annum.
%!
%! years = [1.20 1.40 1.60 2.10 2.30 3.00 3.10 3.30 3.30 3.80 4.00 4.10 ...
%!               4.10 4.20 4.60 5.00 5.20 5.40 6.00 6.10 6.90 7.20 8.00 8.30 ...
%!               8.80 9.10 9.60 9.70 10.40 10.60]';
%! salary = [39 46 38 44 40 57 60 54 64 57 63 56 57 57 61 68 66 83 81 94 92 ...
%!           98 101 114 109 106 117 113 122 122]';
%!
%! bootridge (salary, years);
%! 
%! % We can see from the intercept that the starting starting salary is $25.2 K
%! % and that salary increase per year of experience is $9.4 K.

%!demo
%!
%! % Two-sample unpaired test on independent samples.
%!
%! score = [54 23 45 54 45 43 34 65 77 46 65]';
%! gender = {'male' 'male' 'male' 'male' 'male' 'female' 'female' 'female' ...
%!           'female' 'female' 'female'}';
%!
%! % Difference between means
%! % Note that the 'dim' argument in `bootlm` automatically changes the default
%! % coding to simple contrasts, which are centered.
%! MAT  = bootlm (score, gender, 'nboot', 0, 'display', 'off', ...
%!                'dim', 1, 'posthoc', 'trt_vs_ctrl');
%! bootridge (MAT.Y, MAT.X, 2);
%!
%! % Group means
%! MAT  = bootlm (score, gender, 'nboot', 0, 'display', 'off', 'dim', 1);
%! bootridge (MAT.Y, MAT.X, 2, [], [], MAT.L);

%!demo
%!
%! % One-way repeated measures design.
%! % The data is from a study on the number of words recalled by 10 subjects
%! % for three time condtions, in Loftus & Masson (1994) Psychon Bull Rev. 
%! % 1(4):476-490, Table 2.
%!
%! words = [10 13 13; 6 8 8; 11 14 14; 22 23 25; 16 18 20; ...
%!          15 17 17; 1 1 4; 12 15 17;  9 12 12;  8 9 12];
%! seconds = [1 2 5; 1 2 5; 1 2 5; 1 2 5; 1 2 5; ...
%!            1 2 5; 1 2 5; 1 2 5; 1 2 5; 1 2 5;];
%! subject = [ 1  1  1;  2  2  2;  3  3  3;  4  4  4;  5  5  5; ...
%!             6  6  6;  7  7  7;  8  8  8;  9  9  9; 10 10 10];
%!
%! % Frequentist framework: wild bootstrap of linear model, with orthogonal
%! % polynomial contrast coding followed up with treatment vs control
%! % hypothesis testing.
%! MAT = bootlm (words, {subject,seconds},  'display', 'off', 'varnames', ...
%!                  {'subject','seconds'}, 'model', 'linear', 'contrasts', ...
%!                  'poly', 'dim', 2, 'posthoc', 'trt_vs_ctrl', 'nboot', 0);
%!
%! % Ridge regression and bayesian analysis of posthoc comparisons
%! bootridge (MAT.Y, MAT.X, '*', 200, 0.05, MAT.L);
%!
%! % Frequentist framework: wild bootstrap of linear model, with orthogonal
%! % polynomial contrast coding followed up estimating marginal means.
%! MAT = bootlm (words, {subject,seconds},  'display', 'off', 'nboot', 0, ...
%!                  'model', 'linear', 'contrasts', 'poly', 'dim', 2);
%!
%! % Ridge regression and bayesian analysis of model estimates. Note that group-
%! % mean Bayes Factors are NaN under the flat prior on the intercept whereas
%! % the contrasts we just calculated had proper Normal priors.
%! bootridge (MAT.Y, MAT.X, '*', 200, 0.05, MAT.L);

%!demo
%!
%! % One-way design with continuous covariate. The data is from a study of the
%! % additive effects of species and temperature on chirpy pulses of crickets,
%! % from Stitch, The Worst Stats Text eveR
%!
%! pulse = [67.9 65.1 77.3 78.7 79.4 80.4 85.8 86.6 87.5 89.1 ...
%!          98.6 100.8 99.3 101.7 44.3 47.2 47.6 49.6 50.3 51.8 ...
%!          60 58.5 58.9 60.7 69.8 70.9 76.2 76.1 77 77.7 84.7]';
%! temp = [20.8 20.8 24 24 24 24 26.2 26.2 26.2 26.2 28.4 ...
%!         29 30.4 30.4 17.2 18.3 18.3 18.3 18.9 18.9 20.4 ...
%!         21 21 22.1 23.5 24.2 25.9 26.5 26.5 26.5 28.6]';
%! species = {'ex' 'ex' 'ex' 'ex' 'ex' 'ex' 'ex' 'ex' 'ex' 'ex' 'ex' ...
%!            'ex' 'ex' 'ex' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv' ...
%!            'niv' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv'};
%!
%! % Estimate regression coefficients using 'anova' contrast coding 
%! MAT  = bootlm (pulse, {temp, species}, 'model', 'linear', ...
%!                           'continuous', 1, 'display', 'off', ...
%!                           'contrasts', 'anova', 'nboot', 0);
%!
%! % Ridge regression and bayesian analysis of regression coefficients
%! % MAT.X: column 1 is intercept, column 2 is temp (continuous), column 3 
%! % is species (categorical).
%! bootridge (MAT.Y, MAT.X, 3, 200, 0.05);

%!demo
%!
%! % Variations in design for two-way ANOVA (2x2) with interaction. 
%!
%! % Arousal was measured in rodents assigned to four experimental groups in a
%! % between-subjects design with two factors: group (lesion/control) and
%! % stimulus (fearful/neutral). In this design, each rodent is allocated to one 
%! % combination of levels in group and stimulus, and a single measurment of
%! % arousal is made. The question we are asking here is, does the effect of a
%! % fear-inducing stimulus on arousal depend on whether or not rodents had a
%! % lesion?
%!
%! group = {'control' 'control' 'control' 'control' 'control' 'control' ...
%!          'lesion'  'lesion'  'lesion'  'lesion'  'lesion'  'lesion'  ...
%!          'control' 'control' 'control' 'control' 'control' 'control' ...
%!          'lesion'  'lesion'  'lesion'  'lesion'  'lesion'  'lesion'};
%! 
%! stimulus = {'fearful' 'fearful' 'fearful' 'fearful' 'fearful' 'fearful' ...
%!             'fearful' 'fearful' 'fearful' 'fearful' 'fearful' 'fearful' ...
%!             'neutral' 'neutral' 'neutral' 'neutral' 'neutral' 'neutral' ...
%!             'neutral' 'neutral' 'neutral' 'neutral' 'neutral' 'neutral'};
%!
%! arousal = [0.78 0.86 0.65 0.83 0.78 0.81 0.65 0.69 0.61 0.65 0.59 0.64 ...
%!            0.54 0.6 0.67 0.63 0.56 0.55 0.645 0.565 0.625 0.485 0.655 0.515];
%!
%! % Fit between-subjects design
%! [STATS, BOOTSTAT, AOVSTAT, PRED_ERR, MAT] = bootlm (arousal, ...
%!                                   {group, stimulus}, 'seed', 1, ...
%!                                   'display', 'off', 'contrasts', 'simple', ...
%!                                   'model', 'full', ...
%!                                   'method', 'bayes');
%!
%! % Ridge regression and bayesian analysis of regression coefficients
%! % MAT.X: column 1 is intercept, column 2 is temp (continuous), column 3 
%! % is species (categorical).
%! bootridge (MAT.Y, MAT.X, '*', 200, 0.05);
%!
%! % Now imagine the design is repeated stimulus measurements in each rodent
%! ID = [1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12]';
%!
%! % Fit model including ID as a blocking-factor
%! MAT = bootlm (arousal, {ID, group, stimulus}, 'seed', 1, 'nboot', 0, ...
%!               'display', 'off', 'contrasts', 'simple', 'method', 'bayes', ...
%!               'model', [1 0 0; 0 1 0; 0 0 1; 0 1 1]);
%!
%! % Ridge regression and bayesian analysis of regression coefficients
%! % MAT.X: column 1 is intercept, column 2 is temp (continuous), column 3 
%! % is species (categorical).
%! bootridge (MAT.Y, MAT.X, '*', 200, 0.05);

%!demo
%!
%! % Analysis of nested one-way design.
%!
%! % Nested model example from:
%! % https://www.southampton.ac.uk/~cpd/anovas/datasets/#Chapter2
%!
%! data = [4.5924 7.3809 21.322; -0.5488 9.2085 25.0426; ...
%!         6.1605 13.1147 22.66; 2.3374 15.2654 24.1283; ...
%!         5.1873 12.4188 16.5927; 3.3579 14.3951 10.2129; ...
%!         6.3092 8.5986 9.8934; 3.2831 3.4945 10.0203];
%!
%! clustid = [1 3 5; 1 3 5; 1 3 5; 1 3 5; ...
%!            2 4 6; 2 4 6; 2 4 6; 2 4 6];
%!
%! group = {'A' 'B' 'C'; 'A' 'B' 'C'; 'A' 'B' 'C'; 'A' 'B' 'C'; ...
%!          'A' 'B' 'C'; 'A' 'B' 'C'; 'A' 'B' 'C'; 'A' 'B' 'C'};
%!
%! % Fit model with cluster-based resampling. We are using Bayesian bootstrap
%! % using 'auto' prior, which effectively applies Bessel's correction to the
%! % variance of the bootstrap distribution for the contrasts (trt_vs_ctrl).
%! % Use 'treatment' coding and return regression coefficients since our
%! % intention is to use the posterior distributions from bayesian bootstrap
%! % to calculate the design effect.
%! [STATS, BOOTSTAT, AOVSTAT, PREDERR, MAT] = bootlm (data, {group}, ...
%!      'clustid', clustid, 'seed', 1, 'display', 'off', 'contrasts', ...
%!      'treatment', 'method', 'bayes', 'prior', 'auto');
%!
%! % Or we can get a obtain the design effect empirically using resampling.
%! % We already fit the model accounting for clustering, now lets fit it
%! % under I.I.D. (i.e. without clustering). As above, use 'treatment' coding.
%! [STATS_SRS, BOOTSTAT_SRS] = bootlm (data, {group}, 'seed', 1, 'display', ...
%!      'off', 'contrasts', 'treatment', 'method', 'bayes');
%!
%! % Empirically calculate the design effect averaged over the variance of
%! % of the contrasts we are interested in
%! Var_true = var (BOOTSTAT, 0, 2);
%! Var_iid  = var (BOOTSTAT_SRS, 0, 2);
%! DEFF = mean (Var_true ./ Var_iid);
%! % Or more simply, we can use the deffcalc function, which does the same thing.
%! % We take the mean DEFF across all contrasts for a stable global penalty.
%! DEFF = mean (deffcalc (BOOTSTAT, BOOTSTAT_SRS)) 
%!
%! % Refit the model using orthogonal (helmert) contrasts and a hypothesis
%! % matrix specifying pairwise comparisons. Set nboot to 0 to avoid resampling.
%! MAT = bootlm (data, {group}, 'clustid', clustid, 'display', 'off', ...
%!               'contrasts', 'helmert', 'dim', 1, 'posthoc', 'pairwise', ...
%!               'nboot', 0);
%!
%! % Fit a cluster-robust empirical Bayes model using our bootstrap estimate of
%! % the design effect and using the hypothesis matrix to define the comparisons
%! bootridge (MAT.Y, MAT.X, '*', 200, 0.05, MAT.L, DEFF);
%!
%! % Compare this to using a maximum cluster size as an upperbound for Deff
%! g = max (accumarray (clustid(:), 1, [], @sum));  % g is max. cluster size
%! bootridge (MAT.Y, MAT.X, '*', 200, 0.05, MAT.L, g); % Upperbound DEFF is g
%!
%! % Note: Using the empirical DEFF (~1.5) instead of the upper-bound (4.0) 
%! % recovers inferential power, as seen by the higher Bayes Factor (lnBF10) 
%! % and narrower credible intervals.

%!demo
%!
%! % Generic univariate example with auto-added intercept
%! m = 40;
%! x = linspace (-1, 1, m).';
%! X = x;                          % No intercept column; function will add it
%! beta = [2; 0.5];
%! randn ('twister', 123);
%! y = beta(1) + beta(2) * x + 0.2 * randn (m, 1);
%! % Run bootridge with a small bootstrap count for speed
%! bootridge (y, X, [], 100, 0.05, [], 1, 123);
%! % Please be patient, the calculations will be completed soon...

%!demo
%!
%! % Generic multivariate outcome example with explicit intercept
%! m = 35;
%! x = linspace (-2, 2, m).';
%! X = [ones(m,1), x];
%! B = [1.5,  2.0;    % intercepts for 2 outcomes
%!      0.6, -0.3];   % slopes for 2 outcomes
%! randn ('twister', 123);
%! E = 0.25 * randn (m, 2);
%! Y = X * B + E;
%! % Run bootridge with small bootstrap count
%! bootridge (Y, X, [], 100, 0.10, [], 1, 321);
%! % Please be patient, the calculations will be completed soon...

%!demo
%! %% --- Stress-test: Simulated Large-Scale Patch-seq Project (bootridge) ---
%! %% N = 7500 cells
%! %% p = 15 features
%! %% q = 2000 genes
%! %% This tests memory handling and global lambda optimization.
%!
%! N = 7500;
%! p = 15;
%! q = 2000;
%! nboot = 100;
%!
%! % Set random seeds for the simulation
%! rand ('seed', 123);
%! randn ('seed', 123);
%!
%! fprintf ('Simulate Large-Scale Patch-seq Dataset (%d x %d)...\n', N, q);
%!
%! % Generate design matrix X (E-phys features)
%! X = [ones(N,1), randn(N, p-1)];
%!
%! % Generate sparse multivariate outcome Y (Gene expression)
%! % Approx 120MB of data
%! true_beta = randn (p, q) .* (rand (p, q) > 0.9);
%! 
%! % Set signal-to-noise ratio to 0.5
%! target_snr = 0.5;
%! beta_no_intercept = true_beta(2:end, :);
%! signal_var_per_gene = sum (beta_no_intercept.^2, 1);
%! snr_per_gene = signal_var_per_gene / (0.5^2);
%! current_snr = mean (snr_per_gene);
%! scale = sqrt (target_snr / current_snr);
%! true_beta(2:end, :) = true_beta(2:end, :) * scale;
%!
%! % Introduce correlations
%! n_factors = 10; % 10 latent biological processes
%! latent_X = randn (N, n_factors); 
%! % Each latent factor affects about 10% of genes (sparse correlation)
%! latent_beta = randn (n_factors, q) .* (rand (n_factors, q) > 0.90);
%!
%! % Simulate the data with added correlated noise (0.2 strength)
%! Y = X * true_beta + (latent_X * latent_beta * 0.2) + randn (N, q) * 0.5;
%!
%! fprintf('Running bootridge ...\n');
%! tic;
%! % Use TOL = 0.05 for faster convergence in demo
%! S = bootridge (Y, X, [], nboot, 0.05, [], 1, 123, 0.05);
%! runtime = toc;
%!
%! fprintf ('\n--- Performance Results ---\n');
%! fprintf ('Runtime: %.2f seconds\n', runtime);
%! fprintf ('Optimized Lambda: %.6f\n', S.lambda);
%! fprintf ('Total Iterations: %d\n', S.iter);
%!
%! % Accuracy Check on a random gene
%! target_gene = ceil (rand * q);
%! estimated = S.coefficient(:, target_gene);
%! actual = true_beta(:, target_gene);
%! correlation = corr (estimated, actual);
%!
%! % ROC statistics
%! threshold = 3;                                    % corresponds to BF10 of 20
%! fp = sum (S.lnBF10(true_beta == 0) >  threshold); % false positives
%! tp = sum (S.lnBF10(true_beta ~= 0) >  threshold); % true positives
%! fn = sum (S.lnBF10(true_beta ~= 0) <= threshold); % missed true effects
%! power      = tp / (tp + fn);                      % true positive rate
%! fp_rate    = fp / sum (true_beta(:) == 0);        % false positive rate
%! precision  = tp / (tp + fp);                      % true discovery rate
%!
%! fprintf ('Correlation of estimates for Gene %d: %.4f\n', ...
%!          target_gene, correlation);
%! fprintf ('Number of coefficients: [%s] (Expected: [15 x 2000])\n', ...
%!           num2str (size (S.coefficient)));
%! fprintf ('Number of pairwise correlations: [%s] (Expected: 1999000)\n', ...
%!           num2str (size (S.RTAB, 1)));
%! fprintf ('Positive detections (i.e. discoveries) defined hereon as BF10 > 20');
%! fprintf ('\nFalse positive rate (FPR): %.1f%%\n', fp_rate * 100);
%! fprintf ('Precision (i.e. 1-FDR): %.1f%%\n', precision * 100);
%! fprintf ('Power (i.e. TPR): %.1f%%\n', power * 100);

%!demo
%! %% --- Stress-test: Large-Scale Differential Gene Expression (DGE) Simulation ---
%! %% Scenario: Bulk RNA-seq Case-Control Study (e.g., Disease vs. Healthy)
%! %% N = 300 samples (e.g., 150 Patient / 150 Control)
%! %% p = 50 covariates (e.g., 1 Group Indicator + 49 Technical/PEER factors)
%! %% q = 15000 genes (Simultaneously modeled outcomes)
%! %%
%! %% This demo evaluates the multivariate efficiency of bootridge across
%! %% the typical protein-coding transcriptome size, testing memory handling
%! %% and the speed of global lambda optimization.
%!
%! % 1. Setup Dimensions
%! N = 300;      % Total biological samples (Bulk RNA-seq)
%! p = 50;       % 1 Experimental Group + technical covariates (Age, RIN, Batch)
%! q = 15000;    % Total Genes analyzed in this "chunk"
%! nboot = 100;  % Number of bootstrap resamples
%! % Set random seeds for the simulation
%! rand ('seed', 123);
%! randn ('seed', 123);
%!
%! fprintf (cat (2, 'Simulating DGE Dataset: %d samples, %d genes, %d ', ...
%!                  'covariates...\n'), N, q, p);
%!
%! % 2. Generate Design Matrix X
%! % Column 1: Intercept
%! % Column 2: Group Indicator (0 = Control, 1 = Case)
%! % Columns 3-p: Random technical noise (Covariates/PEER factors)
%! Group = [zeros(N/2, 1); ones(N/2, 1)];
%! Covariates = randn (N, p-2);
%! X = [ones(N, 1), Group, Covariates];
%!
%! % 3. Define Biological Signal (The "True" Log-Fold Changes)
%! % We simulate a realistic DGE profile:
%! % 10% of genes are differentially expressed (hits).
%! true_beta = zeros (p, q);
%! sig_genes = ceil (rand (1, round (q * 0.10)) * q);
%! true_beta(2, sig_genes) = (randn (1, length (sig_genes)) * 2); % Group effect
%!
%! % 4. Generate Expression Matrix Y (Log2 TPM / Counts)
%! % Baseline + Group Effect + Gaussian noise.
%! Baseline = 5 + randn (1, q);
%! Y = repmat (Baseline, N, 1) + X * true_beta + randn (N, q) * 1.2;
%!
%! fprintf ('Running Multivariate bootridge (Shrinkage shared across genes)...\n');
%!
%! tic;
%! % CATEGOR = 2: Treats the Group column as categorical (no variance scaling).
%! % SEED = 123: Ensures reproducible bootstrap sampling.
%! % TOL = 0.05: Convergence tolerance for the golden section search.
%! S = bootridge (Y, X, 2, nboot, 0.05, [], 1, 123, 0.05);
%! runtime = toc;
%!
%! % 5. Display Performance Results
%! fprintf ('\n--- Performance Results ---\n');
%! fprintf ('Runtime: %.2f seconds\n', runtime);
%! fprintf ('Optimized Lambda: %.6f\n', S.lambda);
%!
%! % 6. Accuracy Check
%! % Compare estimated Beta (Group Effect) against the Ground Truth Fold-Changes
%! estimated_fc = S.coefficient(2, :);
%! true_fc = true_beta(2, :);
%! correlation = corr (estimated_fc', true_fc');
%!
%! fprintf ('Correlation of Fold-Changes across %d genes: %.4f\n', ...
%!          q, correlation);
%! fprintf ('Number of coefficients: [%s] (Expected: [50 x 15000])\n', ...
%!           num2str (size (S.coefficient)));
%! fprintf ('Number of pairwise correlations: [%s] (Expected: 112492500)\n', ...
%!           num2str (size (S.RTAB, 1)));

%!demo
%! %% --- Stress-test: High-p Voxel-wise Neural Encoding Simulation ---
%! %% Scenario: Reconstructing Visual Stimuli from fMRI BOLD signals
%! %% N = 500 volumes (samples), p = 8000 voxels, q = 1 stimulus feature
%! %% This demo requires the signal package in octave
%!
%! % 1. Setup Dimensions
%! N = 500; p = 8000; q = 1; nboot = 200;
%! rand ('seed', 123); randn ('seed', 123);
%! fprintf('Simulating fMRI Encoding: %d timepoints, %d voxels...\n', N, p);
%!
%! % 2. Generate Design Matrix X (The Voxels)
%! %    Spatial correlation between voxels (columns) and time points (rows)
%! X_raw = randn (N, p-1);
%! X = [ones(N, 1), filter([0.5 1 0.5], 1, X_raw, [], 2)];
%! X(:,2:end) = filter ([0.1, 0.4, 0.9, 1, 0.6, 0.2], 1, X(:,2:end), [], 1);
%!
%! % 3. Define the "Neural Code" (True Weights)
%! true_beta_sparse = zeros (p, q);                       % Initialise
%! active_voxels = 1 + ceil (rand (1, 25) * (p - 1));     % 50 spatial clusters
%! true_beta_sparse(active_voxels) = randn (25, 1) * 100; % Active voxels
%! kernel = [0.05 0.1 0.4 0.8 1 0.8 0.4 0.1 0.05];        % Smoothing kernel
%! true_beta = filter (kernel, 1, true_beta_sparse);      % Smooth active voxels
%!
%! % 4. Generate outcome Y (The Stimulus)
%! % Signal from smoothed clusters + Gaussian noise
%! Y = X * true_beta + randn (N, q);
%!
%! % 5. Estimate Design Effect (Deff) to account for serial dependence
%! % We use the autocorrelation of Y to estimate the variance inflation factor
%! %
%! %  Bayley, G. V., & Hammersley, J. M. (1946). The "Effective" Number of
%! %    Independent Observations in an Autocorrelated Time Series. Supplement to
%! %    the Journal of the Royal Statistical Society, 8(2), 184–197.
%! %    https://doi.org/10.2307/2983560
%! try
%!   info = ver;
%!   isoctave = any (ismember ({info.Name}, 'Octave'));
%!   if (isoctave)
%!     pkg load signal;
%!   end
%!   [r, lags] = xcov (Y - mean (Y), 10, 'coeff');
%!   Deff = 1 + 2 * sum (r(lags > 0));
%!   fprintf ('Estimated Design Effect (Deff): %.3f\n', Deff);
%! catch
%!   Deff = 1;
%! end
%!
%! % 6. Run bootridge
%! fprintf ('Running bootridge (Global Lambda Optimization)...\n');
%! tic;
%! % Run the bootridge function
%! S = bootridge (Y, X, [], nboot, 0.05, [], Deff, 123, 0.05);
%! runtime = toc;
%!
%! % 7. Performance Results
%! estimated_beta = S.coefficient;
%! correlation = corr (estimated_beta, true_beta);
%! fprintf ('\n--- Performance Results ---\n');
%! fprintf ('Runtime: %.2f seconds\n', runtime);
%! fprintf ('Optimized Lambda: %.6f\n', S.lambda);
%!
%! fprintf ('Correlation of Voxel Weight Map: %.4f\n', correlation);
%! fprintf ('Number of coefficients: [%s] (Expected: [8000 x 1])\n', ...
%!           num2str (size (S.coefficient)));

%!test
%! % Basic functionality: univariate, intercept auto-add, field shapes
%! m = 30;
%! x = linspace (-1, 1, m).';
%! X = x;                           % No intercept provided
%! randn ('twister', 123);
%! y = 1.0 + 0.8 * x + 0.1 * randn (m,1);
%! S = bootridge (y, X, [], 200, 0.05, [], 1.1, 777);
%! % Check expected fields and sizes
%! assert (isfield (S, 'coefficient'));
%! assert (~ isfield (S, 'estimate'));
%! assert (size (S.coefficient, 2) == 1);
%! assert (size (S.coefficient, 1) == 2);     % intercept + slope
%! assert (isfinite (S.lambda) && (S.lambda > 0));
%! assert (isfinite (S.df_lambda) && (S.df_lambda > 0) && ...
%!         (S.df_lambda <= m));
%! assert (all (S.CI_lower(:) <= S.coefficient(:) + eps));
%! assert (all (S.CI_upper(:) + eps >= S.coefficient(:)));
%! assert (isfinite (S.Sigma_Y_hat) && (S.Sigma_Y_hat > 0));
%! assert (iscell (S.Sigma_Beta) && (numel (S.Sigma_Beta) == 1));
%! assert (all (size (S.Sigma_Beta{1}) == [2, 2]));
%! assert (S.nboot == 200);
%! assert (S.Deff == 1.1);

%!test
%! % Hypothesis matrix L: return linear estimate instead of coefficients
%! m = 28;
%! x = linspace (-1.5, 1.5, m).';
%! X = [ones(m,1), x];              % Explicit intercept is first column
%! randn ('twister', 123);
%! y = 3.0 + 0.4 * x + 0.15 * randn (m,1);
%! % Contrast to extract only the slope (second coefficient)
%! L = [0; 1];
%! S = bootridge (y, X, [], 100, 0.10, L, 1, 99);
%! assert (~ isfield (S, 'coefficient'));
%! assert (isfield (S, 'estimate'));
%! assert (all (size (S.estimate) == [1, 1]));
%! assert (all (size (S.CI_lower) == [1, 1]));
%! assert (all (size (S.CI_upper) == [1, 1]));
%! assert (all (size (S.BF10 ) == [1, 1]));
%! assert (iscell (S.prior) && all (size (S.prior) == [1, 1]));

%!test
%! % Categorical predictor supplied via CATEGOR (no scaling)
%! m = 36;
%! % Two-level factor coded as centered +/-0.5 (column 2), plus a continuous
%! g = repmat ([ -0.5; 0.5 ], 18, 1);
%! x = linspace (-2, 2, m).';
%! X = [ones(m,1), g, x];
%! beta = [1.0; 0.7; -0.2];
%! randn ('twister', 123);
%! y = X * beta + 0.25 * randn (m, 1);
%! categor = 2;                  % column 2 is categorical (excludes intercept)
%! S = bootridge (y, X, categor, 100, 0.05, [], 1, 2024);
%! assert (isfield (S, 'coefficient'));
%! assert (size (S.coefficient, 1) == 3);
%! assert (isfinite (S.lambda) && (S.lambda > 0));
%! % Check CI bracketing for all coefficients
%! assert (all (S.CI_lower(:) <= S.coefficient(:) + eps));
%! assert (all (S.CI_upper(:) + eps >= S.coefficient(:)));

%!test
%! % Multivariate outcomes and Deff scaling: Sigma_Y_hat should scale by Deff
%! m = 32;
%! x = linspace (-1, 1, m).';
%! X = [ones(m,1), x];
%! B = [2.0, -1.0; 0.5, 0.8];
%! randn ('twister', 123);
%! Y = X * B + 0.2 * randn (m, 2);
%! S1 = bootridge (Y, X, [], 100, 0.10, [], 1, 42);
%! S2 = bootridge (Y, X, [], 100, 0.10, [], 2, 42);
%! assert (all (size (S1.Sigma_Y_hat) == [2, 2]));
