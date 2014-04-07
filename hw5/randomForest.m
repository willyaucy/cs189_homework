function dTrees=randomForest(XtrainWithLabels, depth)
    treeSize = 23;
    dTrees = zeros(treeSize,1);
    numSamples = size(XtrainWithLabels, 1);
    for i=1:treeSize
        numSamplesSubset = ceil(numSamples * 0.67);
        sampleIndices = ceil( rand(numSamplesSubset,1) .* numSamples );
        XtrainSubset = XtrainWithLabels(sampleIndices, :);
        dTrees(i) = dTree(XtrainSubset, depth, true);
    end