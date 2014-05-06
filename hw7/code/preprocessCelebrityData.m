function data=preprocessCelebrityData(unmasked_pixels)
	data = [];
	filenames = ls('../CelebrityDatabase');
	if size(filenames, 1) == 1 %unix computer
		%filenamelist = regexp(filenames, '\t', 'split');
		%filenames = filenamelist';
        filenames = getAllFiles('../CelebrityDatabase');
	end
	for i=1:size(filenames, 1)
		filename = strtrim(char(filenames(i,:)));
		if size(filename,2) > 4 && isequal(filename(size(filename,2)-3: size(filename,2)), '.jpg')
			path = strcat('../CelebrityDatabase/', filename);
			colorimg = imread(path);
			grayscaleimg = rgb2gray(colorimg);
			im_vector = grayscaleimg(unmasked_pixels);
			data(size(data,1)+1,:) = im_vector';
		end
    end
    
function fileList = getAllFiles(dirName)

  dirData = dir(dirName);      %# Get the data for the current directory
  dirIndex = [dirData.isdir];  %# Find the index for directories
  fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files
  if ~isempty(fileList)
    fileList = cellfun(@(x) fullfile(dirName,x),...  %# Prepend path to files
                       fileList,'UniformOutput',false);
  end
  subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
  validIndex = ~ismember(subDirs,{'.','..'});  %# Find index of subdirectories
                                               %#   that are not '.' or '..'
  for iDir = find(validIndex)                  %# Loop over valid subdirectories
    nextDir = fullfile(dirName,subDirs{iDir});    %# Get the subdirectory path
    fileList = [fileList; getAllFiles(nextDir)];  %# Recursively call getAllFiles
  end