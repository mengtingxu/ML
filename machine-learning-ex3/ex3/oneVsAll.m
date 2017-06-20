function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i
% Some useful variables
m = size(X, 1);
n = size(X, 2);
% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);
% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================

options=optimset('GradObj','on','MaxIter',50);  
  
for k=1:num_labels  
    initial_theta=zeros(n+1,1);%全初始化为可以  
    theta=fmincg(@(t)(lrCostFunction(t,X,(y==k),lambda)),initial_theta,options);  
    %将y=k的变为1 不等于的变为0  进行逻辑回归 返回一组参数theta 
    all_theta(k,:)=theta';%记录所有的theta 输入一个x时 x*每组theta求出hx hx即为这个分类的概率    
end  

% =========================================================================


end
