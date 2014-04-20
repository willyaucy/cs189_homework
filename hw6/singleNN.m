function result=singleNN(dataWithLabel)
    MINI_BATCH_SIZE = 200;
    NUM_EPOCHS = 10;
    NUM_CLASSES = 10;
    numData = size(dataWithLabel, 1);
    numFeatures = size(dataWithLabel, 2) - 1;
    numBatches = ceil(numData/MINI_BATCH_SIZE);
    perm = randperm( size(numData) );
    W = rand(NUM_CLASSES, numFeatures); % NUM_CLASSES by numFeatures
    B = rand(NUM_CLASSES, 1);
    for e=1:NUM_EPOCHS
        dataWithLabel = dataWithLabel(perm, :);
        for i=0:numBatches-1
            for j=1:MINI_BATCH_SIZE
                X = dataWithLabel(i*200 + j, 1:numFeatures)'; % numFeatures by 1
                Y = sigmoid(W*X + B); % NUM_CLASSES by 1
                T = zeros(NUM_CLASSES, 1);
                for k=1:size(X, 2)
                    T(dataWithLabel(numFeatures+1, k), 1) = 1;
                end
                temp = diag(Y.*(1-Y)) * (Y-T); % NUM_CLASSES by 1
                W_grad = W_grad + temp * X'; % NUM_CLASSES by numFeatures
                B_grad = B_grad + temp;
            end
            W = W - alpha * W_grad;
            B = B - alpha * B_grad;
        end
        %{
        for i=0:numBatches-1
            dataWithLabel = dataWithLabel(i*200 + 1, i*200 + 200, :)'; % numFeatures+1 by MINI_BATCH_SIZE
            X = dataWithLabel(1:numFeatures, :); % numFeatures by MINI_BATCH_SIZE
            labels = zeros(NUM_CLASSES, MINI_BATCH_SIZE);
            for j=1:size(X, 2)
                labels(dataWithLabel(numFeatures+1, j), j) = 1;
            end
            Y = sigmoid(W*X); % NUM_CLASSES by MINI_BATCH_SIZE
            W = W + alpha * mean_square_update(Y, labels);
        end
        %}
    end
    
function result=sigmoid(X)
    result = 1/(1+exp(-1 * X));
    
function mean_square_update(Y, labels)
    derivative = Y .* (1-Y); % NUM_CLASSES by MINI_BATCH_SIZE
    derivative * (Y - labels)
    