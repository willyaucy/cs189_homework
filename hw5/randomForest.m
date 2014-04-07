function dTrees=randomForest(XtrainWithLabels, depth, chi)
    treeSize = 15;
    for i=1:treeSize
        dTrees(i).root = 0;
    end
    numSamples = size(XtrainWithLabels, 1);
    for i=1:treeSize
        numSamplesSubset = ceil(numSamples * 0.67);
        sampleIndices = ceil( rand(numSamplesSubset,1) .* numSamples );
        XtrainSubset = XtrainWithLabels(sampleIndices, :);
        dTrees(i).root = dTree(XtrainSubset, depth, true, chi);
    end