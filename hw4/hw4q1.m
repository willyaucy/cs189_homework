function result=hw4q1()
X = [0 3; 1 3; 0 1; 1 1];
Y = [1;1;0;0];
betaRV = [-2 1 0];
l = 0.07; %lagrange multiplier

%preprocessing
X = cat( 2, X, ones( size(X, 1), 1 ) ); %append a column of 1 to X, for the bias

for i=1:2
    fprintf('\ncalculating new mu and betas at iteration %d\n', i);
    clear mu;
    mu = []; %reinitialize mu
    for j=1:size(X, 1)
        fprintf( '\ncalculating mu %d...\n', j );
        mu = cat( 1, mu, getMu( betaRV, X(j,:) ) );
    end
    gradient = getGradient(betaRV, l, X, Y, mu);
    hessian = getHessian(X, l , mu);
    betaRV = newtonsMethod(betaRV, hessian, gradient);
    fprintf( 'mu: ' );
    disp(mu');
    fprintf( 'betaRV: ' );
    disp(betaRV);
    fprintf( 'gradient: ' );
    disp(gradient);
    fprintf( 'hessian: ' );
    disp(hessian);
end

function mu=getMu(betaRV,xRV)
    fprintf( 'Inside getMu()...\n' );
    fprintf( 'betaRV size: %d %d\nxRV size: %d %d\n', size(betaRV), size(xRV) );
    fprintf( 'size of their product: %d %d\n', size(betaRV * xRV') );
    %disp(betaRV);
    %disp(xRV);
    mu = 1/(1+exp(-1*betaRV * xRV'));
    fprintf( 'Finished getMu().\n' );
    
function gradient=getGradient(betaRV, l, X, y, mu)
    fprintf( 'Inside getGradient()...\n' );
    gradient = 2*betaRV'*l - X'*(y-mu);
    fprintf( 'size of the gradient %d %d\n', size(gradient) );
    fprintf( 'Finished getGradient().\n' );

function hessian=getHessian(X, l, mu)
    fprintf( 'Inside getHessian()...\n' );
    W = diag(diag( mu*(1-mu') ));
    hessian = 2*l + X'*W*X;
    fprintf( 'size of Hessian %d %d\n', size(hessian) );
    fprintf( 'Finished getHessian().\n' );
    
function newBetaRV=newtonsMethod(betaRV, hessian, gradient)
    fprintf( 'Inside newtonsMethod()...\n' );
    newBetaRV = (betaRV' - hessian*gradient)';
    fprintf( 'size of new betaRV %d %d\n', size(newBetaRV) );
    fprintf( 'Finished newtonsMethod().\n' );