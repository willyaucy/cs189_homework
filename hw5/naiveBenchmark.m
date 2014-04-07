function result=naiveBenchmark()
    load('spam.mat'); % loads Xtrain, ytrain, Xtest into the workspace
    num_samples = length(ytrain);
    ytrain = double(ytrain);
    XtrainWithLabels = horzcat(Xtrain,ytrain); % combine the labels with the samples, with labels at last column
    dtree = dTree(XtrainWithLabels, 50);
    load('spam.mat'); % loads Xtrain, ytrain, Xtest into the workspace
    numSamples = size(Xtrain,1);
    numError = 0;
    for i=1:numSamples
       ourLabel = spamOrHam(Xtrain(i,:), dtree);
       actualLabel = ytrain(i,1);
       if ourLabel ~= actualLabel
            numError = numError + 1;
       end
    end
    result = (numSamples-numError)/numSamples;