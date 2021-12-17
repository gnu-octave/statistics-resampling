% Basic uninstall script 
% 

copyfile ('PKG_ADD','PKG_ADD.m'); % copy as m-file for backwards compatibility
copyfile ('PKG_DEL','PKG_DEL.m'); % copy as m-file for backwards compatibility
run (fullfile(pwd,'PKG_ADD.m'));
if isoctave
  % Uninstall for Octave
  dirlist = cell(3,1); % dir list needs to be in decreasing order of length
  dirlist{1} = fullfile (pwd,'inst','helper');
  dirlist{2} = fullfile (pwd,'inst','param');
  dirlist{3} = fullfile (pwd,'inst','');
  n = numel(dirlist);
  octaverc = '~/.octaverc';
  if exist(octaverc,'file')
    S = fileread(octaverc);
    [fid, msg] = fopen (octaverc, 'w');
  else
    error('~/.octaverc does not exist');
  end
  comment = regexptranslate ('escape', '% Load statistics-bootstrap package');
  S = regexprep(S,['(\s*)',comment],'');
  for i=1:n
    S = regexprep(S,strcat('(\s*)', regexptranslate ('escape', strcat('\naddpath (''',dirlist{i},''', ''-end'''))),'');
  end
  fwrite (fid,S,'char');
  fclose (fid);
else
  % Assumming uninstall for Matlab instead
  run (fullfile(pwd,'PKG_DEL'));
  if exist('savepath')
    savepath
  else
    % backwards compatibility
    path2rc;
  end
end

% Notify user that uninstall is complete
disp('This statistics-bootstrap package has been uninstalled from this location')

% Clean up
clear dirlist S comment i ii octaverc fid n msg
delete ('PKG_ADD.m');
delete ('PKG_DEL.m');
