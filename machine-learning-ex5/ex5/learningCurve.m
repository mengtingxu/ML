function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================

% ---------------------- Sample Solution ----------------------
%学习曲线是随着m的不同而对J_tain(theta) J_val(theta)的值做运算 所以m的大小是变化的

for i=1:m
    [theta] = trainLinearReg(X(1:i,:), y(1:i), lambda); 
    %求出不同样本容量下的训练的theta值
    hx = X(1:i,:) *theta;
    error_train(i) = sum((hx-y(1:i)).^2)/(2*i); %求出J_train（theta）的值
    hx_val=Xval *theta;
    error_val(i) = sum((hx_val-yval).^2)/(2*size(Xval,1));%求出J_train（theta）的值
end   
% -------------------------------------------------------------

% =========================================================================

end

