function result = hw7q1()
    t = load('../data/train.mat');
    k = [5 10 20];
    data = preprocessMNIST(t.train);
    for i=1:size(k, 2)
        disp(['Clustering with ' num2str(k(i)) ' means']);
        means = kmeans(data, k(i));
        scrsz = get(0,'ScreenSize');
        for m=1:(k(i)/5)
            figure('Position',[scrsz(1) scrsz(2) scrsz(3) scrsz(4)]);
            for j=1:5
                %disp(means(j+5*(m-1)));
                subplot(ceil(5/2),2,j);
                imagesc(reshape(means(j+5*(m-1),:), sqrt(numel(means(j+5*(m-1),:))), sqrt(numel(means(j+5*(m-1),:)))));
                title(['Mean ' num2str(j+5*(m-1))]);
                colorbar;
            end
        end
    end

function data=preprocessMNIST(dataSet)
    numElem = size(dataSet.images, 1) * size(dataSet.images,2);
    numData = size(dataSet.images, 3);
    data = zeros(numData, numElem);
    for i=1:numData
        image = double(reshape(dataSet.images(:,:,i), numElem, 1))';
        image = image / norm(image);
        data(i, :) = image;
    end
    