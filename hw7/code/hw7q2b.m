function hw7q2b()
	NUM_EIGEN = 50;
	load('mask.mat');
	unmasked_pixels = find(mask(:,:,1));
	data = preprocessCelebrityData(unmasked_pixels); % 158 x 17317
	numData = size(data,1);
	rand_data = data(randperm(numData),:);
	chosen5 = rand_data(1:5,:);
	mean_data = data - repmat(mean(data,1), size(data,1), 1);
	mean_chosen5 = chosen5 - repmat(mean(data,1), 5, 1);
	[u,s,v] = svd(mean_data', 0); %econ size
	errors = zeros(NUM_EIGEN, 1);
	for i=1:NUM_EIGEN
		eigenvectors = u(:,1:i); %17317 by i
		result = mean_chosen5 * eigenvectors; %5 by i
		for j=1:5
			scaled_eigenvectors = eigenvectors .* repmat(result(j,:), 17317, 1);
			face = sum(scaled_eigenvectors,2); %17317 by 1 face
			errors(i) = errors(i) + norm(face - mean_chosen5(j)') / 5;
			if i==NUM_EIGEN
				face = face + mean_chosen5(j,:)';
        		full_im = zeros(size(mask(:,:,1)));
				full_im(unmasked_pixels) = normalize_vec(chosen5(j,:)');
				figure('Position',[100 100 600 300]);
        		subplot(1,2,1);
        		imshow(full_im);
        		colormap(gray);
				full_im = zeros(size(mask(:,:,1)));
				full_im(unmasked_pixels) = normalize_vec(face);
        		subplot(1,2,2);
				imshow(full_im);
				colormap(gray);
			end
		end
	end

function vec = normalize_vec(vec)
	vec = vec ./ ( max(vec) - min(vec) );
	vec = vec - min(vec);