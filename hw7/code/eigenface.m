function result = eigenface()
	load('mask.mat');
	unmasked_pixels = find(mask(:,:,1));
	data = preprocessCelebrityData(unmasked_pixels); % 158 x 17317
	mean_data = data - repmat(mean(data,1), size(data,1), 1);
	[u,s,v] = svd(mean_data', 0); %econ size
	for i=1:10
		eigenvector = u(:,i);
		full_im = zeros(size(mask(:,:,1)));
		full_im(unmasked_pixels) = eigenvector;
		figure;
		imagesc(full_im);
		colormap(gray);
	end
	

function data=preprocessCelebrityData(unmasked_pixels)
	data = [];
	filenames = ls('../CelebrityDatabase');
	for i=1:size(filenames, 1)
		filename = strtrim(filenames(i,:));
		if size(filename,2) > 4 && isequal(filename(size(filename,2)-3: size(filename,2)), '.jpg')
			path = strcat('../CelebrityDatabase/', filename);
			colorimg = imread(path);
			grayscaleimg = rgb2gray(colorimg);
			im_vector = grayscaleimg(unmasked_pixels);
			data(size(data,1)+1,:) = im_vector';
		end
	end