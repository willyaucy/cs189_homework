function accuracy = benchmark()
close all;
load('spam.mat'); %loads Xtrain, ytrain into the workspace
num_samples = length(ytrain);
perm = randperm(num_samples);
Xtrain = Xtrain(perm,:);
ytrain = ytrain(perm);
num_folds = 1;
fold_size = floor(num_samples / 10);
accuracy = zeros(num_folds,1);
lambdas = [0.1 0.01 0.5 0.05 2];
rhos = [-1 0.01 0.001 0.002 0.005 0.0001 0.0002 0.00001];
useSDescent = true;

for t=1:2;
    useSDescent = ~useSDescent;
    for method=1:3;
        for i=1:size(lambdas,2);
            for j=1:size(rhos,2);
                if useSDescent %if using stochastic gradient descent
                    numIter = 3; %Actual number of iterations = numIter * number of training samples 
                else
                    numIter = 1000; %Exact number of iterations
                end
                test_upperbound = min(1+fold_size-1, num_samples);
                xtest = Xtrain(1:test_upperbound,:);
                ytest = ytrain(1:test_upperbound);
                xtrain_fold = Xtrain(test_upperbound+1:num_samples,:);
                ytrain_fold = ytrain(test_upperbound+1:num_samples);
                betaRV = train(xtest, ytest, method, lambdas(1,i), useSDescent, rhos(1,j), numIter);
                if isempty(betaRV) == 0 %if it is empty it means we returned incorrectly/intendedly
                    hits = 0;
                    num_tests = length(ytest);
                    for k=1:num_tests;
                        if predictor(xtest(k,:), betaRV, 0) == ytest(k);
                            hits = hits+1;
                        end
                    end
                    fprintf('Accuracy at loop: %d\n', i);
                    fprintf('num_hits: %d / total: %d\n', hits, num_tests);
                    fprintf('Accuracy: %f\n\n', hits / num_tests); %just for tighter output
                    accuracy(floor(i/fold_size)+1) = hits / num_tests;
                end
            end
        end
    end
end

% x: feature vector
% betaRV: it's the beta :P
% bias
% return 1 if x is predicted to be a spam
function prediction=predictor(x, betaRV, bias)
    prediction = 1/(1+exp(betaRV*transpose(x)+bias)) > 0;