function [W_list,B_list]=singleNN(dataWithLabel)
    MINI_BATCH_SIZE = 200;
    NUM_EPOCHS = 500;
    NUM_CLASSES = 10;
    alpha = 0.1;
    numData = size(dataWithLabel, 1);
    numFeatures = size(dataWithLabel, 2) - 1;
    numBatches = ceil(numData/MINI_BATCH_SIZE);
    W_list = zeros(NUM_CLASSES, numFeatures, NUM_EPOCHS/10);
    B_list = zeros(NUM_CLASSES, NUM_EPOCHS/10);
    W = rand(NUM_CLASSES, numFeatures); % NUM_CLASSES by numFeatures
    B = rand(NUM_CLASSES, 1);
    for e=1:NUM_EPOCHS
        fprintf('%d\n',e);
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
                temp = diag(Y.*(1-Y)) * (Y - T); % NUM_CLASSES by 1
                W_grad = W_grad + temp * X'; % NUM_CLASSES by numFeatures
                B_grad = B_grad + temp;
            end
            W = W - alpha* W_grad;
            B = B - alpha* B_grad;
        end
        if mod(e,10) == 0
            W_list(:,:,e/10) = W;
            B_list(:,e/10) = B;
        end
    end
    
function result=sigmoid(X)
    result = 1./(1+exp(-1. * X));
    