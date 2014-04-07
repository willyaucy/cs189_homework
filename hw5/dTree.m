function result=dtree()
    load('spam.mat'); % loads Xtrain, ytrain, Xtest into the workspace
    Xtrain = double(Xtrain); % converts to doubles
    Xtrain = horzcat(Xtrain,ytrain); % combine the labels with the samples, with labels at last column
    root = growTree(Xtrain, 0);
    
function node=growTree(root, depth)
    if depth > 8
        node.left = 0;
        node.right = 0;
        node.attr = 0;
        node.splitpoint = 0;
        return;
    end
    disp(size(root))
    disp(depth);
    numFeatures = size(root, 2)-1;
    nonSpams = root( root(:,numFeatures+1)==0, : ); % matrix of non spams
    spams = root( root(:,numFeatures+1)==1, : ); % matrix of spams
    attrMeansNonSpams = mean(nonSpams, 1); % means of the attributes of the non spams, expected size 1*57
    attrMeansSpams = mean(spams, 1);
    attrMeans = mean([attrMeansNonSpams;attrMeansSpams], 1); % mean of the means
    previousEntropy = calculateEntropy(root); % initial entropy
    maxInfoGain = -Inf;
    bestAttr = 0;
    bestLeftSubtree = [];
    bestRightSubtree = [];
    for i=1:numFeatures
        leftSubtree = root( root(:,i)<attrMeans(i), : ); % data that belongs to left subtree
        rightSubtree = root( root(:,i)>=attrMeans(i), : ); % data that belongs to right subtree
        sizeLeft = size(leftSubtree, 1);
        sizeRight = size(rightSubtree, 1);
        totalSize = sizeLeft+sizeRight;
        infoGain = previousEntropy - ( sizeLeft/totalSize * calculateEntropy(leftSubtree) + sizeRight/totalSize * calculateEntropy(rightSubtree) );
        disp(['inside loop initial entropy: ' num2str(previousEntropy) ' infoGain: ' num2str(infoGain)]);
        if infoGain > maxInfoGain
           disp(' found max ');
           bestAttr = i;
           maxInfoGain = infoGain;
           bestLeftSubtree = leftSubtree;
           bestRightSubtree = rightSubtree;
        end
    end
    node.left = growTree(bestLeftSubtree, depth+1);
    node.right = growTree(bestRightSubtree, depth+1);
    node.attr = bestAttr;
    node.splitpoint = attrMeans(bestAttr);
    
function entropy=calculateEntropy(tree)
    numFeatures = size(tree, 2)-1;
    nonSpams = tree( tree(:,numFeatures+1)==0, : ); % matrix of non spams
    spams = tree( tree(:,numFeatures+1)==1, : ); % matrix of spams
    numNonSpams = size(nonSpams, 1);
    numSpams = size(spams, 1);
    totalSize = numNonSpams + numSpams;
    if numSpams/totalSize == 0 || numNonSpams/totalSize == 0
        entropy = 0;
    else
        entropy = - numSpams/totalSize * log2(numSpams/totalSize) - numNonSpams/totalSize * log2(numNonSpams/totalSize);
    end
    