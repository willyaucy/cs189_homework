function result=singleNNPredictor(W, B, dataWithLabel)
    numData = size(dataWithLabel, 1);
    numFeatures = size(dataWithLabel, 2) - 1;
    X = dataWithLabel(:, 1:numFeatures); % numData by numFeatures
    labels = dataWithLabel(:, numFeatures+1);
    Y = sigmoid(W'*X' + B);  % numClass by numData
    actualLabels = find(Y==max(Y, 1));
    
    
function result=sigmoid(X)
    result = 1/(1+exp(-1 * X));