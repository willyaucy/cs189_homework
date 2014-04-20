function result=singleNNBenchmark()
    load('train_small.mat');
    for i=1:size(train, 2)
        dataWithLabel = preprocessMNIST(train{i});
        classifiers_params = singleNN(dataWithLabel);
        numPoints = size(classifiers_params, 1);
        for j=1:numPoints
           singleNNPredictor(classifiers_params(j, 1), classifiers_params(j, 2), dataWithLabel);
        end
    end
    
function dataWithLabel=preprocessMNIST(dataSet)
    numElem = numel(dataSet.images);
    numData = size(dataSet, 1);
    dataWithLabel = [transpose(double(reshape(dataSet.images, numElem, numData))) dataSet.labels];