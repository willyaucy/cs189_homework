function results = benchmark()
close all;
load('spam.mat'); %loads Xtrain, ytrain into the workspace
num_samples = length(ytrain);
perm = randperm(num_samples);
Xtrain = Xtrain(perm,:);
ytrain = ytrain(perm);
num_folds = 10;
fold_size = floor(num_samples / 10);
accuracy = zeros(num_folds,1);
lambdas = [0.1,0.01,0.5,0.05,2,0.1,0.1,0.1,0.01,0.01,0.01,0.01,0.5,0.5,0.05,0.05,0.05,2,0.01,0.01,0.5,0.5,0.5,0.5,0.05,0.05,0.05,2,2,0.01];
rhos = [0.00001,0.00001,0.00001,0.00001,0.00001,0.0001,0.0002,0.00001,0.002,0.0001,0.0002,0.00001,0.0001,0.00001,0.002,0.0001,0.0002,0.00001,0.002,0.005,0.01,0.001,0.002,0.005,0.01,0.002,0.005,0.002,0.005,0.0002];
methodz = [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2];
useSDescents = [false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,true,true,true,true,true,true,true,true,true,true,true];

standardizedX = standardizeMatCols(Xtrain);
transformedX = transformMat(Xtrain);
binarizedX = binarizeMat(Xtrain);

for i=1:size(lambdas,2);
    method = methodz(1,i);
    useSDescent = useSDescents(1,i);
    if method == 1
        Xtrain = standardizedX;
    elseif method == 2
        Xtrain = transformedX;
    elseif method == 3
        Xtrain = binarizedX;
    end

    if useSDescent %if using stochastic gradient descent
        numIter = 3; %Actual number of iterations = numIter * number of training samples 
    else
        numIter = 5000; %Exact number of iterations
    end
    accuracy = zeros(num_folds,1);
    f = fopen('output.txt', 'a');
    fprintf(f, 'method = %d, lambda = %f, Stochastic = %d, Rho = %f, numIter = %d\r\n', method, lambdas(1,i), useSDescent, rhos(1,i), numIter);
    fclose(f);
    fprintf('method = %d, lambda = %f, Stochastic = %d, Rho = %f, numIter = %d\n', method, lambdas(1,i), useSDescent, rhos(1,i), numIter);
    for f=1:fold_size:num_samples;
        test_upperbound = min(f+fold_size-1, num_samples);
        xtest = Xtrain(f:test_upperbound,:);
        ytest = ytrain(f:test_upperbound);
        xtrain_fold = [Xtrain(1:f-1,:); Xtrain(test_upperbound+1:num_samples,:)];
        ytrain_fold = [ytrain(1:f-1);   ytrain(test_upperbound+1:num_samples)];
        betaRV = train(xtrain_fold, ytrain_fold, lambdas(1,i), useSDescent, rhos(1,i), numIter);
        if isempty(betaRV) == 0 %if it is empty it means we returned incorrectly/intendedly
            hits = 0;
            num_tests = length(ytest);
            for k=1:num_tests;
                if predictor(xtest(k,:), betaRV, 0) == ytest(k);
                    hits = hits+1;
                end
            end
            fprintf(' --- Accuracy at fold %d: %f\n', floor(f/fold_size)+1, hits / num_tests);
            accuracy(floor(f/fold_size)+1) = hits / num_tests;
        end
        %break; %comment this out to get the result for all 10 folds.
    end
    f = fopen('output.txt', 'a');
    fprintf(f, 'Cross-validation accuracy: %f\r\n', mean(accuracy));
    fclose(f);
    fprintf('Cross-validation accuracy: %f\n', mean(accuracy));
end

% x: feature vector
% betaRV: it's the beta :P
% bias
% return 1 if x is predicted to be a spam
function prediction=predictor(x, betaRV, bias)
    prediction = 1/(1+exp(-1*betaRV*transpose(x)+bias)) > 0.5;

function stdMatrix=standardizeMatCols(X)
X = X - repmat( mean(X, 1), size(X, 1), 1 );
X = X ./ repmat( std(X, 0, 1), size(X, 1), 1 );
stdMatrix = X;

function transformedMatrix=transformMat(X)
transformedMatrix = log(X+0.1);

function binarizedMatrix=binarizeMat(X)
binarizedMatrix = (X>0);