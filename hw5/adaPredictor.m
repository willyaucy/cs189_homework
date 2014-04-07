function label=adaPredictor(data, dTrees, alphas)
    weightedLabel = 0;
    for i=1:size(dTrees,2)
        label=spamOrHam(data, dTrees(i).root);
        if label == 0
            label = -1;
        end
        weightedLabel = weightedLabel + alphas(i) * label;
    end
    label = (sign(weightedLabel)+1)/2;
    