function [W_list,B_list,totalLossList]=multiNN(dataWithLabel, crossEntropyOn)
    MINI_BATCH_SIZE = 200;
    NUM_EPOCHS = 500;
    NUM_CLASSES = 10;
    NUM_NODES_HID1 = 300; % number of nodes in hidden layer 1
    NUM_NODES_HID2 = 100;
    alpha = 0.5;
    numData = size(dataWithLabel, 1);
    numFeatures = size(dataWithLabel, 2) - 1;
    numBatches = ceil(numData/MINI_BATCH_SIZE);
    W_list = zeros(NUM_LAYERS, max(numClasses,NUM_NODES_HID1,NUM_NODES_HID2), max(numFeatures,NUM_NODES_HID1,NUM_NODES_HID2), NUM_EPOCHS/10);
    B_list = zeros(NUM_LAYERS, max(numClasses,NUM_NODES_HID1,NUM_NODES_HID2), NUM_EPOCHS/10);
    totalLossList = zeros(NUM_EPOCHS/10,1);
    W1 = rand(NUM_NODES_HID1, numFeatures)-0.5; % NUM_NODES_HID1 by numFeatures
    B1 = rand(NUM_NODES_HID1, 1)-0.5; % NUM_NODES_HID1 by 1
    W2 = rand(NUM_NODES_HID2, NUM_NODES_HID1)-0.5; % NUM_NODES_HID2 by NUM_NODES_HID1
    B2 = rand(NUM_NODES_HID2, 1)-0.5; % NUM_NODES_HID2 by 1
    W3 = rand(NUM_CLASSES, NUM_NODES_HID2)-0.5; % NUM_CLASSES by NUM_NODES_HID2
    B3 = rand(NUM_CLASSES, 1)-0.5; % NUM_CLASSES by 1
    for e=1:NUM_EPOCHS
        fprintf('Epoch %d\n',e);
        totalLoss = 0;
        perm = randperm( numData );
        dataWithLabel = dataWithLabel(perm, :);
        for i=0:numBatches-1
            W1_grad = zeros(NUM_NODES_HID1, numFeatures);
            B1_grad = zeros(NUM_NODES_HID1, 1);
            W2_grad = zeros(NUM_NODES_HID2, NUM_NODES_HID1);
            B2_grad = zeros(NUM_NODES_HID2, 1);
            W3_grad = zeros(NUM_CLASSES, NUM_NODES_HID2);
            B3_grad = zeros(NUM_CLASSES, 1);
            for j=1:min(MINI_BATCH_SIZE, numData - i*MINI_BATCH_SIZE)
                X1 = dataWithLabel(i*MINI_BATCH_SIZE + j, 1:numFeatures)'; % numFeatures by 1
                S1 = getS(W1, X1, B1);
                tanhS1 = tanh(S1);
                S2 = getS(W2, tanhS1, B2);
                tanhS2 = tanh(S2);
                S3 = getS(W3, tanhS2, B3);
                Y = sigmoid(S3); % NUM_CLASSES by 1
                T = zeros(NUM_CLASSES, 1);
                T(dataWithLabel(i*MINI_BATCH_SIZE + j, numFeatures+1)+1) = 1;
                if crossEntropyOn
                    delta3 = diag(Y.*(1-Y)) * (- T./Y + (1-T)./(1-Y));
                    totalLoss = totalLoss + getCrossEntropyLoss(Y, T);
                else
                    delta3 = diag(Y.*(1-Y)) * (Y - T); % NUM_CLASSES by 1
                    totalLoss = totalLoss + getMeanSquareLoss(Y, T);
                end
                W3_grad = W3_grad + delta3 * tanhS2'; % NUM_CLASSES by numFeatures
                B3_grad = B3_grad + delta3;
                delta2 = getTanhDelta(tanhS2, W3, delta3);
                W2_grad = W2_grad + delta2 * tanhS1'; % NUM_CLASSES by numFeatures
                B2_grad = B2_grad + delta2;
                delta1 = getTanhDelta(tanhS1, W2, delta2);
                W1_grad = W1_grad + delta1 * X1'; % NUM_CLASSES by numFeatures
                B1_grad = B1_grad + delta1;
            end
            W3 = W3 - alpha* decay_function(e, NUM_EPOCHS)* W3_grad;
            B3 = B3 - alpha* decay_function(e, NUM_EPOCHS)* B3_grad;
            W2 = W2 - alpha* decay_function(e, NUM_EPOCHS)* W2_grad;
            B2 = B2 - alpha* decay_function(e, NUM_EPOCHS)* B2_grad;
            W1 = W1 - alpha* decay_function(e, NUM_EPOCHS)* W1_grad;
            B1 = B1 - alpha* decay_function(e, NUM_EPOCHS)* B1_grad;
        end
        if mod(e,10) == 0
            W_list(1,1:NUM_NODES_HID1,1:numFeatures,e/10) = W1; % 3 by NUM_NODES_HID1 by numFeatures by NUM_EPOCHS/10
            B_list(1,1:NUM_NODES_HID1,e/10) = B1;
            W_list(2,1:NUM_NODES_HID2,1:NUM_NODES_HID1,e/10) = W2; % NUM_NODES_HID2 by NUM_NODES_HID1
            B_list(2,1:NUM_NODES_HID2,e/10) = B2;
            W_list(3,1:NUM_CLASSES,1:NUM_NODES_HID2,e/10) = W3; % NUM_CLASSES by NUM_NODES_HID2
            B_list(3,1:NUM_CLASSES,e/10) = B3;
            totalLossList(e/10) = totalLoss;
        end
    end
    
function result=sigmoid(X)
    result = 1./(1+exp(-1. * X));
    
function result=getMeanSquareLoss(Y, T)
    result = sum((Y-T).^2)/2;
    
function result=getCrossEntropyLoss(Y, T)
    result = -1*sum( (T.*log(Y)+(1-T).*log(1-Y)) );
    
function decay=decay_function(e, n)
    decay = -1 / (1 + exp(-10*e / n)) + 1;
    
function outputCurrent=getS(weightCurrent, XCurrent, biasCurrent)
    outputCurrent = weightCurrent * XCurrent + biasCurrent;

function backPropDelta=getTanhDelta(tanhCurrent, weightUpper, deltaUpper)
    backPropDelta = (1-tanhCurrent.^2) .* (weightUpper'*deltaUpper);
    