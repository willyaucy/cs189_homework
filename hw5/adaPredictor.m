function label=adaPredictor(data, dTrees, alphas)
    counter = zeros(2,1);
    weightedLabel = 0;
    for i=1:size(dTrees,2)
        label=spamOrHam(data, dTrees(i).root)+1;
        counter(label) = counter(label)+1;
        if counter(2) >= counter(1)
            label = 1;
        else
            label = -1;
        end
        weightedLabel = weightedLabel + alphas(i) * label;
    end
    label = (sign(weightedLabel)+1)/2;
    