function accuracy = benchmark()
close all;
load('spam.mat'); %loads Xtrain, ytrain into the workspace
num_samples = size(ytrain)(1);
perm = randperm(num_samples);
Xtrain = Xtrain(perm,:);
ytrain = ytrain(perm);
num_folds = 10;
fold_size = floor(num_samples / num_folds);
accuracy = zeros(num_folds,1);
for i=1:fold_size:num_samples;
    test_upperbound = min(i+fold_size-1, num_samples);
    xtest = Xtrain(i:test_upperbound,:);
    ytest = ytrain(i:test_upperbound);
    xtrain_fold = [Xtrain(1:i-1,:); Xtrain(test_upperbound+1:num_samples,:)];
    ytrain_fold = [ytrain(1:i-1);   ytrain(test_upperbound+1:num_samples)];
    betaRV = train(xtrain_fold, ytrain_fold);

    hits = 0;
    num_tests = size(ytest)(1);
    for k=1:num_tests;
        if predictor(xtest(k,:), betaRV, 0) == ytest(k);
            hits = hits+1;
        end
    end
    fprintf('Accuracy at loop: %d\n', i)
    fprintf('num_hits: %d / total: %d\n', hits, num_tests)
    hits / num_tests
    accuracy(i) = hits / num_tests;
end


% x: feature vector
% betaRV: it's the beta :P
% bias
% return 1 if x is predicted to be a spam
function prediction=predictor(x, betaRV, bias)
    prediction = 1/(1+exp(betaRV*transpose(x)+bias)) > 0;