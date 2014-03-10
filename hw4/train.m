function betaRV=train(Xtrain, ytrain)
betaRV = zeros(1,size(Xtrain,2))*50;
l = 0.1;
ytrain = double(ytrain);
useSDescent = true;
mu = zeros();
if useSDescent
    numIter = 3;
else
    numIter = 1000;
end

for method=2:2
    figure();
    if method == 1
        X = standardizeMatCols(Xtrain);
    elseif method == 2
        X = transformMat(Xtrain);
    else
        X = binarizeMat(Xtrain);
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
                betaRV = sDescent(betaRV, ytrain(j), mu(j), X(j,:), 0.01);
                yaxis = getNll(betaRV, l, ytrain, mu);
                hold on
                plot((i-1)*size(X, 1)+j,yaxis,'.');
                hold off
            end
        end
        %disp(sparse(mu));
        if useSDescent == false
            gradient = getGradient(betaRV, l, X, ytrain, mu);
            betaRV = bDescent(betaRV, gradient, 0.001);
            yaxis = getNll(betaRV, l, ytrain, mu);
            hold on
            plot(i, yaxis, '.');
            hold off
            %display(betaRV);
        end
    end
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