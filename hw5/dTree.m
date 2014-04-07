function root=dTree(XtrainWithLabels, maxDepth, randomFeatures)
    XtrainWithLabels = double(XtrainWithLabels); % converts to doubles
    root = growTree(XtrainWithLabels, 0, maxDepth, randomFeatures);
    
function node=growTree(root, depth, maxDepth, randomFeatures)
    previousEntropy = calculateEntropy(root); % initial entropy
    if previousEntropy == 0 || depth >= maxDepth
        node.attr = 0;
        node.label = getMajority(root);
        return;
    end
    numFeatures = size(root, 2)-1;
    nonSpams = root( root(:,numFeatures+1)==0, : ); % matrix of non spams
    spams = root( root(:,numFeatures+1)==1, : ); % matrix of spams
    attrMeansNonSpams = mean(nonSpams, 1); % means of the attributes of the non spams, expected size 1*57
    attrMeansSpams = mean(spams, 1);
    attrMeans = mean([attrMeansNonSpams;attrMeansSpams], 1); % mean of the means
    maxInfoGain = -Inf;
    bestAttr = 0;
    bestLeftSubtree = [];
    bestRightSubtree = [];
    numFeatures = 1:numFeatures;
    if randomFeatures
        perm = randperm( size(numFeatures,2) );
        perm = perm( 1: ceil( sqrt( size(numFeatures,2) ) ) );
        numFeatures = perm;
    end
    for i=1:size(numFeatures,2)
        leftSubtree = root( root(:,numFeatures(i))<attrMeans(numFeatures(i)), : ); % data that belongs to left subtree
        rightSubtree = root( root(:,numFeatures(i))>=attrMeans(numFeatures(i)), : ); % data that belongs to right subtree
        sizeLeft = size(leftSubtree, 1);
        sizeRight = size(rightSubtree, 1);
        totalSize = sizeLeft+sizeRight;
        if sizeLeft == 0
            weightedEntropy = sizeRight/totalSize * calculateEntropy(rightSubtree);
        elseif sizeRight == 0
            weightedEntropy = sizeLeft/totalSize * calculateEntropy(leftSubtree);
        else
            weightedEntropy = sizeLeft/totalSize * calculateEntropy(leftSubtree) + sizeRight/totalSize * calculateEntropy(rightSubtree);
        end
        infoGain = previousEntropy - weightedEntropy;
        %disp(['inside loop initial entropy: ' num2str(previousEntropy) ' infoGain: ' num2str(infoGain)]);
        if infoGain > maxInfoGain
           %disp(' found max ');
           bestAttr = numFeatures(i);
           maxInfoGain = infoGain;
           bestLeftSubtree = leftSubtree;
           bestRightSubtree = rightSubtree;
        end
    end
    if maxInfoGain == 0
        node.attr = 0;
        node.label = getMajority(root);
    else
        node.left = growTree(bestLeftSubtree, depth+1, maxDepth, randomFeatures);
        node.right = growTree(bestRightSubtree, depth+1, maxDepth, randomFeatures);
        node.attr = bestAttr;
        node.splitpoint = attrMeans(bestAttr);
    end
    
function entropy=calculateEntropy(tree)
    numFeatures = size(tree, 2)-1;
    nonSpams = tree( tree(:,numFeatures+1)==0, : ); % matrix of non spams
    spams = tree( tree(:,numFeatures+1)==1, : ); % matrix of spams
    numNonSpams = size(nonSpams, 1);
    numSpams = size(spams, 1);
    totalSize = numNonSpams + numSpams;
    %if totalSize==0
    %    entropy = 0;
    if numSpams/totalSize == 0
        entropy = - numNonSpams/totalSize * log2(numNonSpams/totalSize);
    elseif numNonSpams/totalSize == 0
        entropy = - numSpams/totalSize * log2(numSpams/totalSize);
    else
        entropy = - numSpams/totalSize * log2(numSpams/totalSize) - numNonSpams/totalSize * log2(numNonSpams/totalSize);
    end
    
function label=getMajority(tree)
    numFeatures = size(tree, 2)-1;
    nonSpams = tree( tree(:,numFeatures+1)==0, : ); % matrix of non spams
    spams = tree( tree(:,numFeatures+1)==1, : ); % matrix of spams
    numNonSpams = size(nonSpams, 1);
    numSpams = size(spams, 1);
    if numSpams >= numNonSpams
        label = 1;
    else
        label = 0;
    end
    
