% This script checks for operational errors in the functions of this package

lasterror ('reset');

try

  % boot
  boot (3, 20);
  boot (3, 20, false, 1);
  boot (3, 20, true, 1);
  boot (3, 20, [], 1);
  boot (3, 20, true, 1, [30,30,0]);

  % bootknife 
  % bootknife:test:1
  y = randn (20,1); 
  strata = [1;1;1;1;1;1;1;1;1;1;2;2;2;2;2;3;3;3;3;3];
  stats = bootknife (y, 2000, @mean);
  stats = bootknife (y, 2000, 'mean');
  stats = bootknife (y, 2000, {@var,1});
  stats = bootknife (y, 2000, {'var',1});
  stats = bootknife (y, 2000, @mean, [], strata);
  stats = bootknife (y, 2000, {'var',1}, [], strata);
  stats = bootknife (y, 2000, {@var,1}, [], strata, 2);
  stats = bootknife (y, 2000, @mean, .1, strata, 2);
  stats = bootknife (y, 2000, @mean, [.05,.95], strata, 2);
  stats = bootknife (y, [2000,200], @mean, .1, strata, 2);
  stats = bootknife (y, [2000,200], @mean, [.05,.95], strata, 2);
  stats = bootknife (y(1:5), 2000, @mean, .1);
  stats = bootknife (y(1:5), 2000, @mean, [.05,.95]);
  stats = bootknife (y(1:5), [2000,200], @mean, .1);
  stats = bootknife (y(1:5), [2000,200], @mean, [.05,.95]);
  % bootknife:test:2
  Y = randn (20); 
  strata = [1;1;1;1;1;1;1;1;1;1;2;2;2;2;2;3;3;3;3;3];
  stats = bootknife (Y, 2000, @mean);
  stats = bootknife (Y, 2000, 'mean');
  stats = bootknife (Y, 2000, {@var, 1});
  stats = bootknife (Y, 2000, {'var',1});
  stats = bootknife (Y, 2000, @mean, [], strata);
  stats = bootknife (Y, 2000, {'var',1}, [], strata);
  stats = bootknife (Y, 2000, {@var,1}, [], strata, 2);
  stats = bootknife (Y, 2000, @mean, .1, strata, 2);
  stats = bootknife (Y, 2000, @mean, [.05,.95], strata, 2);
  stats = bootknife (Y, [2000,200], @mean, .1, strata, 2);
  stats = bootknife (Y, [2000,200], @mean, [.05,.95], strata, 2);
  stats = bootknife (Y(1:5,:), 2000, @mean, .1);
  stats = bootknife (Y(1:5,:), 2000, @mean, [.05,.95]);
  stats = bootknife (Y(1:5,:), [2000,200], @mean, .1);
  stats = bootknife (Y(1:5,:), [2000,200], @mean, [.05,.95]);
  stats = bootknife (Y, 2000, @(Y) mean(Y(:),1)); % Cluster/block resampling
  % Y(1,end) = NaN; % Unequal clustersize
  %stats = bootknife (Y, 2000, @(Y) mean(Y(:),1,'omitnan'));
  % bootknife:test:3
  y = randn (20,1); x = randn (20,1); X = [ones(20,1), x];
  stats = bootknife ({x,y}, 2000, @cor);
  stats = bootknife ({x,y}, 2000, @cor, [], strata);
  stats = bootknife ({y,x}, 2000, @(y,x) pinv(x)*y); % Could also use @regress
  stats = bootknife ({y,X}, 2000, @(y,X) pinv(X)*y);
  stats = bootknife ({y,X}, 2000, @(y,X) pinv(X)*y, [], strata);
  stats = bootknife ({y,X}, 2000, @(y,X) pinv(X)*y, [], strata, 2);
  stats = bootknife ({y,X}, 2000, @(y,X) pinv(X)*y, [.05,.95], strata);
  
  % bootci
  % bootci:test:1
  y = randn (20, 1); 
  bootci (2000, 'mean', y);
  bootci (2000, @mean, y);
  bootci (2000, @mean, y, 'alpha', 0.1);
  bootci (2000, {'mean', y}, 'alpha', 0.1);
  bootci (2000, {@mean, y}, 'alpha', 0.1);
  bootci (2000, {@mean, y}, 'alpha', 0.1, 'seed', 1);
  bootci (2000, {@mean, y}, 'alpha', 0.1, 'type', 'norm');
  bootci (2000, {@mean, y}, 'alpha', 0.1, 'type', 'per');
  bootci (2000, {@mean, y}, 'alpha', 0.1, 'type', 'basic');
  bootci (2000, {@mean, y}, 'alpha', 0.1, 'type', 'bca');
  bootci (2000, {@mean, y}, 'alpha', 0.1, 'type', 'stud');
  bootci (2000, {@mean, y}, 'alpha', 0.1, 'type', 'stud', 'nbootstd', 100);
  bootci (2000, {@mean, y}, 'alpha', 0.1, 'type', 'cal');
  bootci (2000, {@mean, y}, 'alpha', 0.1, 'type', 'cal', 'nbootcal', 200);
  % bootci:test:2
  Y = randn (20); 
  bootci (2000, 'mean', Y);
  bootci (2000, @mean, Y);
  bootci (2000, @mean, Y, 'alpha', 0.1);
  bootci (2000, {'mean', Y}, 'alpha', 0.1);
  bootci (2000, {@mean, Y}, 'alpha', 0.1);
  bootci (2000, {@mean, Y}, 'alpha', 0.1, 'seed', 1);
  bootci (2000, {@mean, Y}, 'alpha', 0.1, 'type', 'norm');
  bootci (2000, {@mean, Y}, 'alpha', 0.1, 'type', 'per');
  bootci (2000, {@mean, Y}, 'alpha', 0.1, 'type', 'basic');
  bootci (2000, {@mean, Y}, 'alpha', 0.1, 'type', 'bca');
  bootci (2000, {@mean, Y}, 'alpha', 0.1, 'type', 'stud');
  bootci (2000, {@mean, Y}, 'alpha', 0.1, 'type', 'cal');
  % bootci:test:3
  y = randn (20,1); x = randn (20,1); X = [ones(20,1),x];
  bootci (2000, @cor, x, y);
  bootci (2000, @(y,X) pinv(X)*y, y, X);
  bootci (2000, @(y,X) pinv(X)*y, y, X, 'alpha', 0.1);
  bootci (2000, {@(y,X) pinv(X)*y, y, X}, 'alpha', 0.1);
  bootci (2000, {@(y,X) pinv(X)*y, y, X}, 'alpha', 0.1, 'type', 'norm');
  bootci (2000, {@(y,X) pinv(X)*y, y, X}, 'alpha', 0.1, 'type', 'per');
  bootci (2000, {@(y,X) pinv(X)*y, y, X}, 'alpha', 0.1, 'type', 'basic');
  bootci (2000, {@(y,X) pinv(X)*y, y, X}, 'alpha', 0.1, 'type', 'bca');
  bootci (2000, {@(y,X) pinv(X)*y, y, X}, 'alpha', 0.1, 'type', 'stud');
  bootci (2000, {@(y,X) pinv(X)*y, y, X}, 'alpha', 0.1, 'type', 'cal');
  
  % bootstrp
  y = randn (20,1);
  bootstat = bootstrp (50, @mean, y);
  
  % bootnhst 
  % bootnhst:test:1
  y = [111.39 110.21  89.21  76.64  95.35  90.97  62.78;
       112.93  60.36  92.29  59.54  98.93  97.03  79.65;
        85.24 109.63  64.93  75.69  95.28  57.41  75.83;
       111.96 103.40  75.49  76.69  77.95  93.32  78.70];
  g = [1 2 3 4 5 6 7;
       1 2 3 4 5 6 7;
       1 2 3 4 5 6 7;
       1 2 3 4 5 6 7];
  p = bootnhst (y(:),g(:),'ref',1,'nboot',[1000,0],'DisplayOpt',false);
  p = bootnhst (y(:),g(:),'nboot',[1000,0],'DisplayOpt',false);
  % bootnhst:test:2
  y = [54       43
       23       34 
       45       65
       54       77
       45       46
      NaN       65];
  g = {'male' 'female'
       'male' 'female'
       'male' 'female'
       'male' 'female'
       'male' 'female'
       'male' 'female'};
  p = bootnhst (y(:),g(:),'ref','male','nboot',[1000,0],'DisplayOpt',false);
  p = bootnhst (y(:),g(:),'nboot',[1000,0],'DisplayOpt',false);
  % bootnhst:test:3
  y = [54  87  45
       23  98  39
       45  64  51
       54  77  49
       45  89  50
       47 NaN  55];
  g = [ 1   2   3
        1   2   3
        1   2   3
        1   2   3
        1   2   3
        1   2   3];
  p = bootnhst (y(:),g(:),'nboot',[1000,0],'DisplayOpt',false);
  p = bootnhst (y(:),g(:),'bootfun',@(y)std(y,1),'DisplayOpt',false);
  p = bootnhst (y(:),g(:),'bootfun',{@std,1},'DisplayOpt',false);
  % bootnhst:test:4
  Y = randn (20, 2); g = [zeros(10, 1); ones(10, 1)];
  func = @(M) cor (M(:,1), M(:,2));
  p = bootnhst (Y, g, 'bootfun', func, 'DisplayOpt', false);

  % bootmode
  % bootmode:test:1
  x=[0.060;0.064;0.064;0.065;0.066;0.068;0.069;0.069;0.069;0.069;0.069;0.069;0.069;0.070;0.070;0.070;
  0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.070;
  0.070;0.070;0.070;0.070;0.070;0.070;0.070;0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071;
  0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.071;0.072;0.072;0.072;0.072;0.072;
  0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;
  0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.072;0.073;0.073;0.073;0.073;0.073;
  0.073;0.073;0.073;0.073;0.073;0.073;0.074;0.074;0.074;0.074;0.074;0.074;0.074;0.074;0.074;0.074;
  0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;0.075;
  0.075;0.075;0.075;0.075;0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076;0.076;
  0.076;0.076;0.076;0.076;0.076;0.076;0.077;0.077;0.077;0.077;0.077;0.077;0.077;0.077;0.077;0.077;
  0.077;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;
  0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.078;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;
  0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;
  0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;0.079;
  0.079;0.079;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;
  0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.080;
  0.080;0.080;0.080;0.080;0.080;0.080;0.080;0.081;0.081;0.081;0.081;0.081;0.081;0.081;0.081;0.081;
  0.081;0.081;0.081;0.081;0.081;0.081;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082;
  0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.082;0.083;0.083;0.083;0.083;0.083;0.083;0.083;0.084;
  0.084;0.084;0.085;0.085;0.086;0.086;0.087;0.088;0.088;0.089;0.089;0.089;0.089;0.089;0.089;0.089;
  0.089;0.089;0.089;0.090;0.090;0.090;0.090;0.090;0.090;0.090;0.090;0.090;0.091;0.091;0.091;0.092;
  0.092;0.092;0.092;0.092;0.093;0.093;0.093;0.093;0.093;0.093;0.094;0.094;0.094;0.095;0.095;0.096;
  0.096;0.096;0.097;0.097;0.097;0.097;0.097;0.097;0.097;0.098;0.098;0.098;0.098;0.098;0.099;0.099;
  0.099;0.099;0.099;0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100;0.100;
  0.100;0.100;0.101;0.101;0.101;0.101;0.101;0.101;0.101;0.101;0.101;0.102;0.102;0.102;0.102;0.102;
  0.102;0.102;0.102;0.103;0.103;0.103;0.103;0.103;0.103;0.103;0.104;0.104;0.105;0.105;0.105;0.105;
  0.105;0.106;0.106;0.106;0.106;0.107;0.107;0.107;0.108;0.108;0.108;0.108;0.108;0.108;0.108;0.109;
  0.109;0.109;0.109;0.109;0.109;0.109;0.110;0.110;0.110;0.110;0.110;0.110;0.110;0.110;0.110;0.110;
  0.110;0.111;0.111;0.111;0.111;0.112;0.112;0.112;0.112;0.112;0.114;0.114;0.114;0.115;0.115;0.115;
  0.117;0.119;0.119;0.119;0.119;0.120;0.120;0.120;0.121;0.122;0.122;0.123;0.123;0.125;0.125;0.128;
  0.129;0.129;0.129;0.130;0.131];
  [H, P, CRITVAL] = bootmode (x, 1, 2000, 'Gaussian');
  [H, P, CRITVAL] = bootmode (x, 2, 2000, 'Gaussian');

  % bootwild
  % bootwild:test:1
  H0 = 150;
  heights = [183, 192, 182, 183, 177, 185, 188, 188, 182, 185].';
  stats = bootwild (heights-H0);
  stats = bootwild (heights-H0, ones(10,1));
  stats = bootwild (heights-H0, [], 2);
  stats = bootwild (heights-H0, [], [1;1;1;1;1;2;2;2;2;2]);
  stats = bootwild (heights-H0, [], [], 2000);
  stats = bootwild (heights-H0, [], [], [], 0.05);
  stats = bootwild (heights-H0, [], [], [], [0.025, 0.975]);
  stats = bootwild (heights-H0, [], [], [], [], 1);
  stats = bootwild (heights-H0, [], [], [], [], []);
  [stats,bootstat] = bootwild(heights-H0);
  % bootwild:test:2
  X = [ones(43,1),...
      [01,02,03,04,05,06,07,08,09,10,11,...
       12,13,14,15,16,17,18,19,20,21,22,...
       23,25,26,27,28,29,30,31,32,33,34,...
       35,36,37,38,39,40,41,42,43,44]'];
  y = [188.0,170.0,189.0,163.0,183.0,171.0,185.0,168.0,173.0,183.0,173.0,...
      173.0,175.0,178.0,183.0,192.4,178.0,173.0,174.0,183.0,188.0,180.0,...
      168.0,170.0,178.0,182.0,180.0,183.0,178.0,182.0,188.0,175.0,179.0,...
      183.0,192.0,182.0,183.0,177.0,185.0,188.0,188.0,182.0,185.0]';
  stats = bootwild (y, X);
  stats = bootwild (y, X, 4);
  stats = bootwild (y, X, [], 2000);
  stats = bootwild (y, X, [], [], 0.05);
  stats = bootwild (y, X, [], [], [0.025, 0.975]);
  stats = bootwild (y, X, [], [], [], 1);
  stats = bootwild (y, X, [], [], [], []);
  [stats, bootstat] = bootwild (y, X);
    
  % bootbayes
  % bootbayes:test:1
  heights = [183, 192, 182, 183, 177, 185, 188, 188, 182, 185].';
  stats = bootbayes (heights);
  stats = bootbayes (repmat (heights, 1, 5));
  stats = bootbayes (heights, ones (10, 1));
  stats = bootbayes (heights, [], 2);
  stats = bootbayes (heights, [], [1;1;1;1;1;2;2;2;2;2]);
  stats = bootbayes (heights, [], [], 2000);
  stats = bootbayes (heights, [], [], [], 0.05);
  stats = bootbayes (heights, [], [], [], [0.025, 0.975]);
  stats = bootbayes (heights, [], [], [], []);
  stats = bootbayes (heights, [], [], [], [], [], []);
  [stats,bootstat] = bootbayes (heights);
  % bootbayes:test:2
  X = [ones(43,1),...
      [01,02,03,04,05,06,07,08,09,10,11,...
       12,13,14,15,16,17,18,19,20,21,22,...
       23,25,26,27,28,29,30,31,32,33,34,...
       35,36,37,38,39,40,41,42,43,44]'];
  y = [188.0,170.0,189.0,163.0,183.0,171.0,185.0,168.0,173.0,183.0,173.0,...
      173.0,175.0,178.0,183.0,192.4,178.0,173.0,174.0,183.0,188.0,180.0,...
      168.0,170.0,178.0,182.0,180.0,183.0,178.0,182.0,188.0,175.0,179.0,...
      183.0,192.0,182.0,183.0,177.0,185.0,188.0,188.0,182.0,185.0]';
  stats = bootbayes (y, X);
  stats = bootbayes (y, X, 4);
  stats = bootbayes (y, X, [], 2000);
  stats = bootbayes (y, X, [], [], 0.05);
  stats = bootbayes (y, X, [], [], [0.025, 0.975]);
  stats = bootbayes (y, X, [], []);
  [stats, bootstat] = bootbayes (y, X);

  % bootlm
  % bootlm:test:1
  % Two-sample unpaired test on independent samples (equivalent to Student's
  % t-test). Note that the absolute value of t-statistic can be obtained by
  % taking the square root of the reported F statistic. In this example,
  % t = sqrt (1.44) = 1.20.
  score = [54 23 45 54 45 43 34 65 77 46 65]';
  gender = {'male' 'male' 'male' 'male' 'male' 'female' 'female' 'female' ...
            'female' 'female' 'female'}';
 
  STATS = bootlm (score, gender, 'display', 'off', 'varnames', 'gender');

  % bootlm:test:2
  % Two-sample paired test on dependent or matched samples equivalent to a
  % paired t-test. As for the first example, the t-statistic can be obtained by
  % taking the square root of the reported F statistic.
  score = [4.5 5.6; 3.7 6.4; 5.3 6.4; 5.4 6.0; 3.9 5.7]';
  treatment = {'before' 'after'; 'before' 'after'; 'before' 'after';
               'before' 'after'; 'before' 'after'}';
  subject = {'GS' 'GS'; 'JM' 'JM'; 'HM' 'HM'; 'JW' 'JW'; 'PS' 'PS'}';
 
  STATS = bootlm (score(:), {treatment(:), subject(:)}, ...
                             'model', 'linear', 'display', 'off', ...
                             'varnames', {'treatment', 'subject'});

  % bootlm:test:3
  % One-way design on the data from a study on the strength of structural beams,
  % in Hogg and Ledolter (1987) Engineering Statistics. New York: MacMillan
  strength = [82 86 79 83 84 85 86 87 74 82 ...
             78 75 76 77 79 79 77 78 82 79]';
  alloy = {'st','st','st','st','st','st','st','st', ...
           'al1','al1','al1','al1','al1','al1', ...
           'al2','al2','al2','al2','al2','al2'}';
  STATS = bootlm (strength, alloy, 'display', 'off', 'varnames', 'alloy');

  % bootlm:test:4
  % One-way repeated measures design on the data from a study on the number of
  % words recalled by 10 subjects for three time condtions, in Loftus & Masson
  % (1994) Psychon Bull Rev. 1(4):476-490, Table 2. Note that the interaction
  % between seconds x subject was dropped from the full model by assigning
  % subject as a random predictor.
  words = [10 13 13; 6 8 8; 11 14 14; 22 23 25; 16 18 20; ...
           15 17 17; 1 1 4; 12 15 17;  9 12 12;  8 9 12];
  seconds = [1 2 5; 1 2 5; 1 2 5; 1 2 5; 1 2 5; ...
             1 2 5; 1 2 5; 1 2 5; 1 2 5; 1 2 5;];
  subject = [ 1  1  1;  2  2  2;  3  3  3;  4  4  4;  5  5  5; ...
              6  6  6;  7  7  7;  8  8  8;  9  9  9; 10 10 10];
  STATS = bootlm (words(:), {seconds(:), subject(:)}, ...
                             'model', 'linear', 'display', 'off', ...
                             'varnames', {'seconds', 'subject'});

  % bootlm:test:5
  % Balanced two-way design with interaction on the data from a study of popcorn
  % brands and popper types, in Hogg and Ledolter (1987) Engineering Statistics.
  % New York: MacMillan
  popcorn = [5.5, 4.5, 3.5; 5.5, 4.5, 4.0; 6.0, 4.0, 3.0; ...
             6.5, 5.0, 4.0; 7.0, 5.5, 5.0; 7.0, 5.0, 4.5];
  brands = {'Gourmet', 'National', 'Generic'; ...
            'Gourmet', 'National', 'Generic'; ...
            'Gourmet', 'National', 'Generic'; ...
            'Gourmet', 'National', 'Generic'; ...
            'Gourmet', 'National', 'Generic'; ...
            'Gourmet', 'National', 'Generic'};
  popper = {'oil', 'oil', 'oil'; 'oil', 'oil', 'oil'; 'oil', 'oil', 'oil'; ...
            'air', 'air', 'air'; 'air', 'air', 'air'; 'air', 'air', 'air'};
 
  STATS = bootlm (popcorn(:), {brands(:), popper(:)}, ...
                             'display', 'off', 'model', 'full', ...
                             'varnames', {'brands', 'popper'});

  % bootlm:test:6
  % Unbalanced two-way design (2x2) on the data from a study on the effects of
  % gender and having a college degree on salaries of company employees,
  % in Maxwell, Delaney and Kelly (2018): Chapter 7, Table 15
  salary = [24 26 25 24 27 24 27 23 15 17 20 16, ...
            25 29 27 19 18 21 20 21 22 19]';
  gender = {'f' 'f' 'f' 'f' 'f' 'f' 'f' 'f' 'f' 'f' 'f' 'f'...
            'm' 'm' 'm' 'm' 'm' 'm' 'm' 'm' 'm' 'm'}';
  degree = [1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 0 0 0 0 0 0 0]';
 
  STATS = bootlm (salary, {gender, degree}, 'model', 'full', ...
                             'display', 'off', 'varnames', ...
                             {'gender', 'degree'});

  % bootlm:test:7
  % Unbalanced two-way design (3x2) on the data from a study of the effect of
  % adding sugar and/or milk on the tendency of coffee to make people babble,
  % in from Navarro (2019): 16.10
  sugar = {'real' 'fake' 'fake' 'real' 'real' 'real' 'none' 'none' 'none' ...
           'fake' 'fake' 'fake' 'real' 'real' 'real' 'none' 'none' 'fake'}';
  milk = {'yes' 'no' 'no' 'yes' 'yes' 'no' 'yes' 'yes' 'yes' ...
          'no' 'no' 'yes' 'no' 'no' 'no' 'no' 'no' 'yes'}';
  babble = [4.6 4.4 3.9 5.6 5.1 5.5 3.9 3.5 3.7...
            5.6 4.7 5.9 6.0 5.4 6.6 5.8 5.3 5.7]';
 
  STATS = bootlm (babble, {sugar, milk}, 'model', 'full', 'display', 'off', ...
                                         'varnames', {'sugar', 'milk'});

  % bootlm:test:8
  % Unbalanced three-way design (3x2x2) on the data from a study of the effects
  % of three different drugs, biofeedback and diet on patient blood pressure,
  % adapted* from Maxwell, Delaney and Kelly (2018): Chapter 8, Table 12
  % * Missing values introduced to make the sample sizes unequal to test the
  %   calculation of different types of sums-of-squares
  drug = {'X' 'X' 'X' 'X' 'X' 'X' 'X' 'X' 'X' 'X' 'X' 'X' ...
          'X' 'X' 'X' 'X' 'X' 'X' 'X' 'X' 'X' 'X' 'X' 'X';
          'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' ...
          'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y' 'Y';
          'Z' 'Z' 'Z' 'Z' 'Z' 'Z' 'Z' 'Z' 'Z' 'Z' 'Z' 'Z' ...
          'Z' 'Z' 'Z' 'Z' 'Z' 'Z' 'Z' 'Z' 'Z' 'Z' 'Z' 'Z'};
  feedback = [1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;
              1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;
              1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0];
  diet = [0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1;
          0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1;
          0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1];
  BP = [170 175 165 180 160 158 161 173 157 152 181 190 ...
        173 194 197 190 176 198 164 190 169 164 176 175;
        186 194 201 215 219 209 164 166 159 182 187 174 ...
        189 194 217 206 199 195 171 173 196 199 180 NaN;
        180 187 199 170 204 194 162 184 183 156 180 173 ...
        202 228 190 206 224 204 205 199 170 160 NaN NaN];
  STATS = bootlm (BP(:), {drug(:), feedback(:), diet(:)}, ...
                                     'model', 'full', ...
                                     'display', 'off', ...
                                     'varnames', {'drug', 'feedback', 'diet'});

  % bootlm:test:9
  % Balanced three-way design (2x2x2) with one of the predictors being a
  % blocking factor. The data is from a randomized block design study on the
  % effects of antioxidant treatment on glutathione-S-transferase (GST) levels
  % in different mouse strains, from Festing (2014), ILAR Journal 55(3):427-476.
  measurement = [444 614 423 625 408  856 447 719 ...
                 764 831 586 782 609 1002 606 766]';
  strain= {'NIH','NIH','BALB/C','BALB/C','A/J','A/J','129/Ola','129/Ola', ...
           'NIH','NIH','BALB/C','BALB/C','A/J','A/J','129/Ola','129/Ola'}';
  treatment={'C' 'T' 'C' 'T' 'C' 'T' 'C' 'T' 'C' 'T' 'C' 'T' 'C' 'T' 'C' 'T'}';
  block = [1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2]';
  STATS = bootlm (measurement/10, {strain, treatment, block}, ...
                             'model', [1 0 0; 0 1 0; 0 0 1; 1 1 0], ...
                             'varnames', {'strain', 'treatment', 'block'}, ...
                             'display', 'off');

  % bootlm:test:10
  % One-way design with continuous covariate on data from a study of the
  % additive effects of species and temperature on chirpy pulses of crickets,
  % from Stitch, The Worst Stats Text eveR
  pulse = [67.9 65.1 77.3 78.7 79.4 80.4 85.8 86.6 87.5 89.1 ...
           98.6 100.8 99.3 101.7 44.3 47.2 47.6 49.6 50.3 51.8 ...
           60 58.5 58.9 60.7 69.8 70.9 76.2 76.1 77 77.7 84.7]';
  temp = [20.8 20.8 24 24 24 24 26.2 26.2 26.2 26.2 28.4 ...
          29 30.4 30.4 17.2 18.3 18.3 18.3 18.9 18.9 20.4 ...
          21 21 22.1 23.5 24.2 25.9 26.5 26.5 26.5 28.6]';
  species = {'ex' 'ex' 'ex' 'ex' 'ex' 'ex' 'ex' 'ex' 'ex' 'ex' 'ex' ...
             'ex' 'ex' 'ex' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv' ...
             'niv' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv' 'niv'};
  STATS = bootlm (pulse, {species, temp}, 'model', 'linear', ...
                            'continuous', 2, 'display', 'off', ...
                            'varnames', {'species', 'temp'});

  % bootlm:test:11
  % Factorial design with continuous covariate on data from a study of the
  % effects of treatment and exercise on stress reduction score after adjusting
  % for age. Data from R datarium package).
  score = [95.6 82.2 97.2 96.4 81.4 83.6 89.4 83.8 83.3 85.7 ...
           97.2 78.2 78.9 91.8 86.9 84.1 88.6 89.8 87.3 85.4 ...
           81.8 65.8 68.1 70.0 69.9 75.1 72.3 70.9 71.5 72.5 ...
           84.9 96.1 94.6 82.5 90.7 87.0 86.8 93.3 87.6 92.4 ...
           100. 80.5 92.9 84.0 88.4 91.1 85.7 91.3 92.3 87.9 ...
           91.7 88.6 75.8 75.7 75.3 82.4 80.1 86.0 81.8 82.5]';
  treatment = {'yes' 'yes' 'yes' 'yes' 'yes' 'yes' 'yes' 'yes' 'yes' 'yes' ...
               'yes' 'yes' 'yes' 'yes' 'yes' 'yes' 'yes' 'yes' 'yes' 'yes' ...
               'yes' 'yes' 'yes' 'yes' 'yes' 'yes' 'yes' 'yes' 'yes' 'yes' ...
               'no'  'no'  'no'  'no'  'no'  'no'  'no'  'no'  'no'  'no'  ...
               'no'  'no'  'no'  'no'  'no'  'no'  'no'  'no'  'no'  'no'  ...
               'no'  'no'  'no'  'no'  'no'  'no'  'no'  'no'  'no'  'no'}';
  exercise = {'lo'  'lo'  'lo'  'lo'  'lo'  'lo'  'lo'  'lo'  'lo'  'lo'  ...
              'mid' 'mid' 'mid' 'mid' 'mid' 'mid' 'mid' 'mid' 'mid' 'mid' ...
              'hi'  'hi'  'hi'  'hi'  'hi'  'hi'  'hi'  'hi'  'hi'  'hi'  ...
              'lo'  'lo'  'lo'  'lo'  'lo'  'lo'  'lo'  'lo'  'lo'  'lo'  ...
              'mid' 'mid' 'mid' 'mid' 'mid' 'mid' 'mid' 'mid' 'mid' 'mid' ...
              'hi'  'hi'  'hi'  'hi'  'hi'  'hi'  'hi'  'hi'  'hi'  'hi'}';
  age = [59 65 70 66 61 65 57 61 58 55 62 61 60 59 55 57 60 63 62 57 ...
         58 56 57 59 59 60 55 53 55 58 68 62 61 54 59 63 60 67 60 67 ...
         75 54 57 62 65 60 58 61 65 57 56 58 58 58 52 53 60 62 61 61]';
  STATS = bootlm (score, {treatment, exercise, age}, ...
                             'model', [1 0 0; 0 1 0; 0 0 1; 1 1 0], ...
                             'continuous', 3, 'display', 'off', ...
                             'varnames', {'treatment', 'exercise', 'age'});

  % bootlm:test:12
  % Unbalanced one-way design with custom, orthogonal contrasts. The statistics
  % relating to the contrasts are shown in the table of model parameters, and
  % can be retrieved from the STATS.coeffs output.
  dv =  [ 8.706 10.362 11.552  6.941 10.983 10.092  6.421 14.943 15.931 ...
         22.968 18.590 16.567 15.944 21.637 14.492 17.965 18.851 22.891 ...
         22.028 16.884 17.252 18.325 25.435 19.141 21.238 22.196 18.038 ...
         22.628 31.163 26.053 24.419 32.145 28.966 30.207 29.142 33.212 ...
         25.694 ]';
  g = [1 1 1 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 3 3 3 ...
       4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5]';
  C = [ 0.4001601  0.3333333  0.5  0.0
        0.4001601  0.3333333 -0.5  0.0
        0.4001601 -0.6666667  0.0  0.0
       -0.6002401  0.0000000  0.0  0.5
       -0.6002401  0.0000000  0.0 -0.5];
  STATS = bootlm (dv, g, 'contrasts', C, 'varnames', 'score', ...
                           'alpha', 0.05, 'display', false);
  STATS = bootlm (dv, g, 'contrasts', C, 'varnames', 'score', ...
                           'alpha', 0.05, 'display', false, 'dim', 1);

  fprintf('Tests completed successfully.\n')

catch

  fprintf('Tests were unsuccessful.\n')

  rethrow(lasterror)

end
