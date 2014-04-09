function label=adaPredictor(data, dTrees, alphas, tfinal)
    weightedLabel = 0;
    for i=1:tfinal
        label=spamOrHam(data, dTrees(i).root);
        if label == 0
            label = -1;
        end
        weightedLabel = weightedLabel + alphas(i) * label;
    end
    label = (sign(weightedLabel)+1)/2;