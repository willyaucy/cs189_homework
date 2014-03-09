function betaRV=hw4q2()
load('spam.mat'); %loads Xtrain, ytrain into the workspace
betaRV=train(Xtrain, ytrain)