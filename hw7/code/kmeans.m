function means = kmeans(data, k)
    numData = size(data, 1);
    numFeatures = size(data, 2) - 1;
    maxValues = max(data(:,1:numFeatures), 1); %row vec of maxValues of each feature
    minValues = min(data(:,1:numFeatures), 1);
    means = rand(k, numFeatures) .* repmat(maxValues, k, 1) + repmat(minValues, k, 1); %initialize means (random choice)
    prevMean = zeros(k, numFeatures);
    counts = zeros(1,k);
    while ~isequal(prevMean, means)
        prevMean = means;
        c = zeros(k, numFeatures); %a set of points belonging to cluster i, where 1 < i < k
        for j=1:numData
            pt = data(j, 1:numFeatures);
            pts = repmat(pt, k, 1); %a col vector of replicated points
            l2Distances = pdist([pts means]);
            [val, index] = min(l2Distances);
            c(index, :) = c(index, :) + pt;
            counts(1,index) = counts(1,index) + 1;
        end
        means = c ./ counts;
    end