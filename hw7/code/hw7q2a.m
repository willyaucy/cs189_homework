function hw7q2a()
	load('mask.mat');
	unmasked_pixels = find(mask(:,:,1));
	data = preprocessCelebrityData(unmasked_pixels); % 158 x 17317
	mean_data = data - repmat(mean(data,1), size(data,1), 1);
	[u,s,v] = svd(mean_data', 0); %econ size
	figure('Position',[100 100 1500 300]);
	for i=1:10
		eigenvector = u(:,i);
		full_im = zeros(size(mask(:,:,1)));
		full_im(unmasked_pixels) = normalize_vec(eigenvector);
		subplot(1,5,mod(i-1,5)+1);
		imshow(full_im);
		colormap(gray);
		if i==5
			figure('Position',[100 100 1500 300]);
		end
	end

function vec = normalize_vec(vec)
	vec = vec ./ ( max(vec) - min(vec) );
	vec = vec - min(vec);