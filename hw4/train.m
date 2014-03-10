function betaRV=train(Xtrain, ytrain, method, l, useSDescent, rho, numIter)
betaRV = ones(1,size(Xtrain,2))*0.1;
ytrain = double(ytrain);
mu = zeros();

%{
%parameters that you can change
l = 0.7; %lagrange multiplier
method = 2; %preprocessing methods
useSDescent = true; %set false if use batch gradient descent, true if stochastic gd
if useSDescent %if using stochastic gradient descent
    rho = 0.0001;
    numIter = 2; %Actual number of iterations = numIter * number of training samples 
else
    rho = 0.01;
    numIter = 1000; %Exact number of iterations
end
%}
changeRho = false; %default not changing the rho in each iteration
realNumIter = numIter;
if useSDescent
    realNumIter = numIter * size(Xtrain,1);
    if rho == -1
        changeRho = true;
    end
end

if changeRho == true || rho ~= -1
    figure();
    title( sprintf('Method %d, lambda = %f, Stochastic = %d, Rho = %f, numIter = %d, changingRHO = %d', method, l, useSDescent, rho, realNumIter, changeRho) );
    fprintf('\nMethod %d, lambda = %f, Stochastic = %d, Rho = %f, numIter = %d, changingRHO = %d:\n', method, l, useSDescent, rho, realNumIter, changeRho);
    if method == 1
        X = standardizeMatCols(Xtrain);
        disp('Preprocessing by standardizing matrix...\n');
    elseif method == 2
        X = transformMat(Xtrain);
        disp('Preprocessing by transforming matrix...\n');
    else
        X = binarizeMat(Xtrain);
        disp('Preprocessing by binarizing matrix...\n');
    end
    for i=1:numIter
        %fprintf('\ncalculating new mu and betas at iteration %d\n', i);
        %clear mu;
        %mu = []; %reinitialize mu
        if useSDescent == false
            for k=1:size(X, 1)
                    mu(k, 1) = getMu( betaRV, X(k,:) );
            end
        else
            for j=1:size(X, 1)
                for k=1:size(X, 1)
                    mu(k, 1) = getMu( betaRV, X(k,:) );
                end
                betaRV = sDescent(betaRV, ytrain(j), mu(j), X(j,:), rho);
                yaxis = getNll(betaRV, l, ytrain, mu);
                hold on
                plot((i-1)*size(X, 1)+j,yaxis,'.');
                hold off
            end
        end
        %disp(sparse(mu));
        if useSDescent == false
            gradient = getGradient(betaRV, l, X, ytrain, mu);
            betaRV = bDescent(betaRV, gradient, rho);
            yaxis = getNll(betaRV, l, ytrain, mu);
            hold on
            plot(i, yaxis, '.');
            hold off
            %display(betaRV);
        end
    end
else
    betaRV = [];
end

function stdMatrix=standardizeMatCols(X)
X = X - repmat( mean(X, 1), size(X, 1), 1 );
X = X ./ repmat( std(X, 0, 1), size(X, 1), 1 );
stdMatrix = X;

function transformedMatrix=transformMat(X)
transformedMatrix = log(X+0.1);

function binarizedMatrix=binarizeMat(X)
binarizedMatrix = (X>0);

function mu=getMu(betaRV,xRV)
    %fprintf( 'Inside getMu()...\n' );
    %fprintf( 'betaRV size: %d %d\nxRV size: %d %d\n', size(betaRV), size(xRV) );
    %fprintf( 'size of their product: %d %d\n', size(betaRV * xRV') );
    %disp(betaRV);
    %disp(xRV);
    mu = 1/(1+exp(-1*betaRV * xRV'));
    %fprintf( 'Finished getMu().\n' );

function gradient=getGradient(betaRV, l, X, y, mu)
    %fprintf( 'Inside getGradient()...\n' );
    %fprintf( 'size of betaRV' );
    %disp(size(betaRV));
    %fprintf( 'size of X' );
    %disp(size(X));
    %fprintf( 'size of mu' );
    %disp(size(mu));
    %fprintf( 'size of y' );
    %disp(size(y));
    gradient = 2*norm(betaRV)*l - X'*(y-mu);
    %fprintf( 'size of the gradient %d %d\n', size(gradient) );
    %fprintf( 'Finished getGradient().\n' );
    
function newBetaRV=bDescent(betaRV, gradient, rho)
    %fprintf( 'Inside bDescent()...\n' );
    %fprintf( 'size of the gradient %d %d\nequals?\n', size(gradient) );
    %fprintf( 'size of the betas row vector %d %d\n', size(betaRV') );
    newBetaRV = (betaRV' - rho*gradient)';
    %fprintf( 'Finished bDescent().\n' );
    
function newBetaRV=sDescent(betaRV, y, mu, xRV, rho)
    %{
    fprintf( 'Inside sDescent()...\n' );
    fprintf( 'Inside getGradient()...\n' );
    fprintf( 'size of betaRV' );
    disp(size(betaRV));
    fprintf( 'mu' );
    disp(mu);
    fprintf( 'y' );
    disp(y);
    %}
    newBetaRV = betaRV + rho*(y-mu)*xRV;
    %fprintf( 'Finished sDescent().\n' );
    
function regularizedNll=getNll(betaRV, l, y, mu)
    %fprintf( 'Inside getNll()...\n' );
    %fprintf( 'size of betaRV is %d %d\n', size(betaRV) );
    %fprintf( 'size of y is %d %d\n', size(y) );
    %fprintf( 'size of mu is %d %d\n', size(mu) );
    %disp(mu(1460,1));
    %disp(mu(1572,1));
    %disp(mu(1650,1));
    %disp('first half');
    %disp((y.*log(mu))');
    %disp('second half');
    %disp(((1-y).*log(1-mu))');
    %disp((y.*log(mu))');
    %disp(sum(isnan(y.*log(mu))));
    %disp(sum(isnan((1-y).*log(1-mu))));
    result = (y.*log(mu)) + ((1-y).*log(1-mu));
    %disp( sparse(result') );
    loss = l*norm(betaRV)^2;
    ll = sum(y.*log(mu)+(1-y).*log(1-mu));
    regularizedNll = loss - ll;
    %disp(regularizedNll);
    %fprintf( 'Finished getNll().\n' );