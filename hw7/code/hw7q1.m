function result = hw7q1()
    t = load('../data/train_small.mat');
    k = [5 10 20];
    dataWithLabel = preprocessMNIST(t.train{1});
    for i=size(k, 2)
        kmeans(dataWithLabel, k);
    end

function data=preprocessMNIST(dataSet)
    numElem = size(dataSet.images, 1) * size(dataSet.images,2);
    numData = size(dataSet.images, 3);
    data = zeros(numData, numElem+1);
    for i=1:numData
        image = double(reshape(dataSet.images(:,:,i), numElem, 1))';
        image = image / norm(image);
        data = image;
    end