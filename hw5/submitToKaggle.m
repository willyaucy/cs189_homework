function labels=submitToKaggle()
    load('spam.mat');
    ytrain = double(ytrain);
    XtrainWithLabels = horzcat(Xtrain, ytrain);
    dTrees = randomForest(XtrainWithLabels, 27, false);
    labels = zeros(size(Xtest,1));
    for i=1:size(Xtest,1)
       labels(i) = forestPredictor(Xtest(i,:), dTrees);
       f = fopen('output.txt', 'a');
       fprintf(f, '%d,%d\n', i, labels(i));
       fclose(f);
    end
    
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