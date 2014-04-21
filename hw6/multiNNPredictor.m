function accuracy = multiNNPredictor(multiLayerW, multiLayerB, dataWithLabel)
    NUM_NODES_HID1 = 300; % number of nodes in hidden layer 1
    NUM_NODES_HID2 = 100;
    NUM_CLASSES = 10;
    numData = size(dataWithLabel, 1);
    numFeatures = size(dataWithLabel, 2) - 1;
    correct = 0;
    W1 = squeeze(multiLayerW(1, 1:NUM_NODES_HID1, 1:numFeatures));
    W2 = squeeze(multiLayerW(2, 1:NUM_NODES_HID2, 1:NUM_NODES_HID1));
    W3 = squeeze(multiLayerW(3, 1:NUM_CLASSES, 1:NUM_NODES_HID2));
    B1 = multiLayerB(1, 1:NUM_NODES_HID1)';
    B2 = multiLayerB(2, 1:NUM_NODES_HID2)';
    B3 = multiLayerB(3, 1:NUM_CLASSES)';
    for i=1:numData
    	X1 = dataWithLabel(i, 1:numFeatures)'; % numFeatures by 1
        label = dataWithLabel(i, numFeatures+1);
        S1 = getS(W1, X1, B1);
        tanhS1 = tanh(S1);
        S2 = getS(W2, tanhS1, B2);
        tanhS2 = tanh(S2);
        S3 = getS(W3, tanhS2, B3);
        Y = sigmoid(S3); % NUM_CLASSES by 1
    	[val, index] = max(Y);
    	if index - 1 == label
    		correct = correct + 1;
    	end
    end
    accuracy = correct / numData;

function outputCurrent=getS(weightCurrent, XCurrent, biasCurrent)
    outputCurrent = weightCurrent * XCurrent + biasCurrent;