function [dtrees,alphas,tfinal]=adaboost(XtrainWithLabels, depth)
    numSamples = size(XtrainWithLabels, 1);
    numFeatures = size(XtrainWithLabels, 2)-1;
    samples = zeros(numSamples,numFeatures+1);
    alphas = zeros(20,1);
    dist = ones(numSamples,1)./numSamples; % initialize as uniform distribution
    tfinal = 0;
    for t=1:100
        dtrees(t).root = 0;
    end
    for t=1:100 % number of rounds
        fprintf('%d\n', t);
        for i=1:numSamples
            ind = sampleFromDistribution(dist);
            samples(i,:) = XtrainWithLabels(ind,:);
        end
        dtrees(t).root = dTree(samples, depth, false, false);
        [error, correctClassifications] = getError(XtrainWithLabels, dtrees(t).root, dist);
        if error >= 0.5
            tfinal = t-1;
            return;
        end
        alphas(t,1) = getAlpha(error);
        dist = updateDist(dist, alphas(t,1), correctClassifications);
    end
    tfinal = t;

function ind=sampleFromDistribution(dist)
    randNum = rand();
    currVal = 0;
    for i=1:size(dist,1)
        currVal = currVal + dist(i);
        if randNum <= currVal
            ind = i;
            return;
        end
    end
    ind = size(dist,1); 
    
function [error, correctClassifications,labels]=getError(XtestWithLabels, dtree, dist)
    numSamples = size(XtestWithLabels,1);
    numFeatures = size(XtestWithLabels,2)-1;
    error = 0;
    correctClassifications = zeros(numSamples,1);
    for i=1:numSamples
       ourLabel = spamOrHam(XtestWithLabels(i,:), dtree);
       actualLabel = XtestWithLabels(i,numFeatures+1);
       if ourLabel ~= actualLabel
            error = error + dist(i);
            correctClassifications(i) = -1;
       else
            correctClassifications(i) = 1;
       end
    end
    
function alpha=getAlpha(error)
    alpha = log( (1-error)/error ) / 2;
    
function dist=updateDist(dist, alpha, correctClassifications)
    dist = dist.*exp(correctClassifications.*-alpha);
    dist = dist./sum(dist);
       