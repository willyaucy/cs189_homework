function label=forestPredictor(data, dTrees)
    counter = zeros(2,1);
    for i=1:size(dTrees,1)
        label=spamOrHam(data, dTrees(i));
        counter(label) = counter(label)+1;
    end
    if counter(1) >= counter(0)
        label = 1;
    else
        label = 0;
    end