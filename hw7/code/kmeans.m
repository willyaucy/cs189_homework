function result = kmeans(data, k)
    numData = size(data, 1);
    numFeatures = size(data, 2) - 1;
    maxValues = max(data(:,1:numFeatures), 1); %row vec of maxValues of each feature
    minValues = min(data(:,1:numFeatures), 1);
    means = rand(k, numFeatures) .* maxValues + minValues; %initialize means (random choice)
    prevMean = zeros(k, numFeatures);
    while ~isequal(prevMean, means)
        prevMean = means;
        c = zeros(k, numData); %a set of points belonging to cluster i, where 1 < i < k
        for j=1:numData
            pt = data(j, 1:numFeatures);
            pts = repmat(pt, k, 1); %a col vector of replicated points
            l2Distances = pdist(pts, means);
            [val, index] = min(l2Distances);
            c(index, j) = pt;
        end
        means = mean(c, 2);
    end
    
    