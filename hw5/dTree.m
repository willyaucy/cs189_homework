function result=dtree()
    load('spam.mat'); % loads Xtrain, ytrain, Xtest into the workspace
    Xtrain = double(Xtrain); % converts to doubles
    Xtrain = horzcat(Xtrain,ytrain); % combine the labels with the samples, with labels at last column
    root = growTree(Xtrain);
    
function node=growTree(root)
    numFeatures = size(root, 2);
    nonSpams = root( root(:,numFeatures+1)==0, : ); % matrix of non spams
    spams = root( root(:,numFeatures+1)==1, : ); % matrix of spams
    numNonSpams = size(nonSpams, 1);
    numSpams = size(spams, 1);
    attrMeansNonSpams = mean(nonSpams, 1); % means of the attributes of the non spams, expected size 1*57
    attrMeansSpams = mean(spams, 1);
    attrMeans = mean([attrMeansNonSpams;attrMeansSpams], 1); % mean of the means
    previousEntropy = calculateEntropy(root); % initial entropy
    maxInfoGain = -Inf;
    bestAttr = 0;
    for i=1:numFeatures
        leftSubtree = root( root(:,i)<attrMeans(i), : ); % data that belongs to left subtree
        rightSubtree = root( root(:,i)>=attrMeans(i), : ); % data that belongs to right subtree
        sizeLeft = size(leftSubtree, 1);
        sizeRight = size(rightSubtree, 1);
        totalSize = sizeLeft+sizeRight;
        infoGain = previousEntropy - ( sizeLeft/totalSize * calculateEntropy(leftSubtree) + sizeRight/totalSize * calculateEntropy(rightSubtree) );
        if infoGain > maxInfoGain
           bestAttr = i;
           maxInfoGain = infoGain;
           bestLeftSubtree = leftSubtree;
           bestRightSubtree = rightSubtree;
        end
    end
    node.left = growTree(bestLeftSubtree);
    node.right = growTree(bestRightSubtree);
    node.attr = bestAttr;
    node.splitpoint = attrMeans(bestAttr);
    
function entropy=calculateEntropy(tree)
    numFeatures = size(tree, 2);
    nonSpams = tree( tree(:,numFeatures+1)==0, : ); % matrix of non spams
    spams = tree( tree(:,numFeatures+1)==1, : ); % matrix of spams
    numNonSpams = size(nonSpams, 1);
    numSpams = size(spams, 1);
    totalSize = numNonSpams + numSpams;
    entropy = - numSpams/totalSize * log2(numSpams/totalSize) - numNonSpams/totalSize * log2(numNonSpams/totalSize);