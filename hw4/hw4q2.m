function newBetaRV=hw4q2(betaRV)
close all;
load('spam.mat'); %loads Xtrain, ytrain into the workspace

Xtrain = binarizeMat(Xtrain);
lambda = 0.1;
rho = 0.001;
useSDescent = false;

if isempty(betaRV)
    newBetaRV=train(Xtrain, ytrain, lambda, useSDescent, rho, 1000);
end

Xtest = binarizeMat(Xtest);
hits = 0;
num_tests = length(Xtest);
for k=1:num_tests;
    fprintf( '%d,%d\n', k, predictor(Xtest(k,:), newBetaRV, 0) );
end


function binarizedMatrix=binarizeMat(X)
    binarizedMatrix = (X>0);

function prediction=predictor(x, betaRV, bias)
    prediction = 1/(1+exp(-1*betaRV*transpose(x)+bias)) > 0.5;