function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training eXamples
J_history = zeros(num_iters, 1);
t = zeros(size(theta));
k = zeros(size(theta));
for iter = 1:num_iters
    h = X*theta;
    for j = 1:size(X,2)
        k(j) = alpha*(1/m)*sum((h-y).*X(:,j));
        t(j) = theta(j) - k(j);
    end
    
    for j = 1:size(X,2)
        theta(j) = t(j);
    end
    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
