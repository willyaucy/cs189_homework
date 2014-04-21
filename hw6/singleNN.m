function [W_list,B_list, totalLossList]=singleNN(dataWithLabel, crossEntropyOn)
    MINI_BATCH_SIZE = 200;
    NUM_EPOCHS = 500;
    NUM_CLASSES = 10;
    alpha = 0.5;
    numData = size(dataWithLabel, 1);
    numFeatures = size(dataWithLabel, 2) - 1;
    numBatches = ceil(numData/MINI_BATCH_SIZE);
    W_list = zeros(NUM_CLASSES, numFeatures, NUM_EPOCHS/10);
    B_list = zeros(NUM_CLASSES, NUM_EPOCHS/10);
    totalLossList = zeros(NUM_EPOCHS/10,1);
    W = rand(NUM_CLASSES, numFeatures)-0.5; % NUM_CLASSES by numFeatures
    B = rand(NUM_CLASSES, 1)-0.5;
    for e=1:NUM_EPOCHS
        fprintf('Epoch %d\n',e);
        totalLoss = 0;
        perm = randperm( numData );
        dataWithLabel = dataWithLabel(perm, :);
        for i=0:numBatches-1
            W_grad = zeros(NUM_CLASSES, numFeatures);
            B_grad = zeros(NUM_CLASSES, 1);
            for j=1:min(MINI_BATCH_SIZE, numData - i*MINI_BATCH_SIZE)
                X = dataWithLabel(i*MINI_BATCH_SIZE + j, 1:numFeatures)'; % numFeatures by 1
                Y = sigmoid(W*X + B); % NUM_CLASSES by 1
                T = zeros(NUM_CLASSES, 1);
                T(dataWithLabel(i*MINI_BATCH_SIZE + j, numFeatures+1)+1) = 1;
                if crossEntropyOn
                    temp = diag(Y.*(1-Y)) * (- T./Y + (1-T)./(1-Y));
                    totalLoss = totalLoss + getCrossEntropyLoss(Y, T);
                else
                    temp = diag(Y.*(1-Y)) * (Y - T); % NUM_CLASSES by 1
                    totalLoss = totalLoss + getMeanSquareLoss(Y, T);
                end
                W_grad = W_grad + temp * X'; % NUM_CLASSES by numFeatures
                B_grad = B_grad + temp;
            end
            W = W - alpha* decay_function(e, NUM_EPOCHS)* W_grad;
            B = B - alpha* decay_function(e, NUM_EPOCHS)* B_grad;
        end
        if mod(e,10) == 0
            W_list(:,:,e/10) = W;
            B_list(:,e/10) = B;
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
