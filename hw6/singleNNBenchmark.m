function singleNNBenchmark()
    load('data/train_small.mat');
    load('data/test.mat');
    data = test;
    for i=1:size(train, 2)
        fprintf('Set %d\n',i);
        dataWithLabel = preprocessMNIST(train{i});
        [W_list,B_list]= singleNN(dataWithLabel);
        numPoints = size(W_list, 3);
        accuracies = size(numPoints, 1);
        totalLoss = size(numPoints, 1);
        for j=1:numPoints
            [accuracies(j), totalLoss(j)] = singleNNPredictor(W_list(:,:,j), B_list(:,j), dataWithLabel);
        end
        scrsz = get(0,'ScreenSize');
        figure('Position',[scrsz(1) scrsz(2) scrsz(3) scrsz(4)]);
        subplot(1,2,1);
        plot(accuracies*100, '-xr');
        title(['Classification Accuracies on Testing Set ' num2str(i)]);
        subplot(1,2,2);
        plot(totalLoss, '-xb');
        title(['Total Training Error on Training Set ' num2str(i)]);
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