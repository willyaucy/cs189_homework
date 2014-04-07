function label=forestPredictor(data, dTrees)
    counter = zeros(2,1);
    for i=1:size(dTrees,2)
        label=spamOrHam(data, dTrees(i).root)+1;
        counter(label) = counter(label)+1;
    end
    if counter(2) >= counter(1)
        label = 1;
    else
        label = 0;
    end