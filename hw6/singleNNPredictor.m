function accuracy = singleNNPredictor(W, B, dataWithLabel)
    numData = size(dataWithLabel, 1);
    numFeatures = size(dataWithLabel, 2) - 1;
    correct = 0;
    for i=1:numData
    	X = dataWithLabel(i, 1:numFeatures)'; % numData by numFeatures
    	label = dataWithLabel(i, numFeatures+1);
    	Y = sigmoid(W*X + B);
    	[val, index] = max(Y);
    	if index - 1 == label
    		correct = correct + 1;
    	end
    end
    accuracy = correct / numData;
    
    
function result=sigmoid(X)
    result = 1./(1+exp(-1. * X));