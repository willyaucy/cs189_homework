function means = kmeans(data, k)
    numData = size(data, 1);
    numFeatures = size(data, 2);
    means = data(randperm(numData),:);
    means = means(1:k,:);
    prevMean = zeros(k, numFeatures);
    while ~isequal(prevMean, means)
        counts = zeros(k,1);
        prevMean = means;
        c = zeros(k, numFeatures); %a set of points belonging to cluster i, where 1 < i < k
        for j=1:numData
            pt = data(j, :);
            pts = repmat(pt, k, 1); %a col vector of replicated points
            l2Distances = sqrt(sum((pts - means).^2, 2));
            [val, index] = min(l2Distances);
            c(index, :) = c(index, :) + pt;
            counts(index) = counts(index) + 1;
        end
        means = c ./ repmat(counts, 1, numFeatures);
        counts
        for k_new=1:k
            if counts(k_new) == 0
                means(k_new,:) = data(ceil(rand()*numData),:);
            end
        end
    end