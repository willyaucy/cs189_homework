function [accuracies, totalLoss] = multiNNBenchmark(crossEntropyOn)
    testTrainingData = false;
    if testTrainingData
        t = load('data/train.mat');
    else
        t = load('data/train_small.mat');
    end
    load('data/test.mat');
    if testTrainingData
        dataWithLabel = preprocessMNIST(t.train);
        [W_list, B_list, totalLoss]= multiNN(dataWithLabel, crossEntropyOn);
        numPoints = size(W_list, 4);
        accuracies = size(numPoints, 1);
        for j=1:numPoints
            accuracies(j) = multiNNPredictor(W_list(:,:,:,j), B_list(:,:,j), dataWithLabel);
        end
        scrsz = get(0,'ScreenSize');
        figure('Position',[scrsz(1) scrsz(2) scrsz(3) scrsz(4)]);
        subplot(1,2,1);
        plot(accuracies*100, '-xr');
        title(['Classification Accuracies on Training Set']);
        subplot(1,2,2);
        plot(totalLoss, '-xb');
        title(['Total Training Error on Training Set']);
    else
        dataWithLabel = preprocessMNIST(t.train{7}); %train with set 7
        [W_list,B_list,totalLoss]= multiNN(dataWithLabel, crossEntropyOn);
        numPoints = size(W_list, 4);
        accuracies = size(numPoints, 1);
        for j=1:numPoints
            accuracies(j) = multiNNPredictor(W_list(:,:,:,j), B_list(:,:,j), preprocessMNIST(test));
        end
        scrsz = get(0,'ScreenSize');
        figure('Position',[scrsz(1) scrsz(2) scrsz(3) scrsz(4)]);
        subplot(1,2,1);
        plot(accuracies*100, '-xr');
        title('Classification Accuracies on Testing Set');
        subplot(1,2,2);
        plot(totalLoss, '-xb');
        title('Total Training Error on Testing Set');
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