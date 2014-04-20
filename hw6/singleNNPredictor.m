function [accuracy, totalLoss]=singleNNPredictor(W, B, dataWithLabel)
    numData = size(dataWithLabel, 1);
    numFeatures = size(dataWithLabel, 2) - 1;
    correct = 0;
    totalLoss = 0;
    for i=1:numData
    	X = dataWithLabel(i, 1:numFeatures)'; % numData by numFeatures
    	label = dataWithLabel(i, numFeatures+1);
    	Y = sigmoid(W*X + B);
    	[val, index] = max(Y);
        T = zeros(size(Y,1),1);
        T(index,1) = 1;
        totalLoss = totalLoss + getMeanSquareLoss(Y,T);
    	if index - 1 == label
    		correct = correct + 1;
    	end
    end
    accuracy = correct / numData;
    
    
function result=sigmoid(X)
    result = 1./(1+exp(-1. * X));
    
function result=getMeanSquareLoss(Y, T)
    result = sum((Y-T).^2)/2;
    
function result=getCrossEntropyLoss(Y, T)
    result = 0; %TODO: implement this