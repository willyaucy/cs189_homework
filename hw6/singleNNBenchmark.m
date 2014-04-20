function singleNNBenchmark()
    load('data/train_small.mat');
    for i=1:size(train, 2)
        dataWithLabel = preprocessMNIST(train{i});
        [W_list,B_list]= singleNN(dataWithLabel);
        numPoints = size(W_list, 3);
        accuracies = size(numPoints, 1);
        for j=1:numPoints
            accuracies(j) = singleNNPredictor(W_list(:,:,j), B_list(:,j), dataWithLabel);
        end
        accuracies
    end
    
function dataWithLabel=preprocessMNIST(dataSet)
    numElem = size(dataSet.images, 1) * size(dataSet.images,2);
    numData = size(dataSet.images, 3);
    dataWithLabel = zeros(numData, numElem+1);
    for i=1:numData
        image = double(reshape(dataSet.images(:,:,i), numElem, 1))';
        image = image / norm(image);
        dataWithLabel(i,:) = [image dataSet.labels(i)];
    end