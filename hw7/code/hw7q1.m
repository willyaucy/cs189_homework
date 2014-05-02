function result = hw7q1()
    t = load('../data/train_small.mat');
    k = [5 10 20];
    data = preprocessMNIST(t.train{1});
    for i=size(k, 2)
        means = kmeans(data, k(i));
        for j=1:k(i)
            scrsz = get(0,'ScreenSize');
            figure('Position',[scrsz(1) scrsz(2) scrsz(3) scrsz(4)]);
            subplot(ceil(k/2),2,j);
            imagesc(reshape(means(j), numel(means), 1));
            title(['Mean ' j]);
        end
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