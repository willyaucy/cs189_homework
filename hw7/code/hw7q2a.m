function hw7q2a()
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
