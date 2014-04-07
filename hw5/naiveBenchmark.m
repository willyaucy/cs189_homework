function result=naiveBenchmark()
    dtree = dTree();
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