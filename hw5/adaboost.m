function [dtrees,alphas]=adaboost(XtrainWithLabels, depth)
    numSamples = size(XtrainWithLabels, 1);
    numFeatures = size(XtrainWithLabels, 2)-1;
    samples = zeros(numSamples,numFeatures+1);
    alphas = zeros(500,1);
    dist = ones(numSamples,1)./numSamples; % initialize as uniform distribution
    for t=1:500
        dtrees(t).root = 0;
    end
    for t=1:500 % number of rounds
        cumdist = cumsum(dist);
        for i=1:numSamples
            ind = ceil( rand()*numSamples );
            if ind == 0
                ind = 1;
            end
            samples(i,:) = XtrainWithLabels(ind,:);
        end
        dtrees(t).root = dTree(samples, depth, false, false);
        [error, correctClassifications] = getError(samples, dtrees(t).root, dist);
        alphas(t,1) = getAlpha(error);
        dist = updateDist(dist, alphas(t,1), correctClassifications);
    end
    
    
function [error, correctClassifications]=getError(XtestWithLabels, dtree, dist)
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
       