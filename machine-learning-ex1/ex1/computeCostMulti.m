function J = computeCostMulti(X, y, theta)
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
predictions =X*theta;  % predictions is hypothesis on all m examples
sqrerrors = (predictions-y).^2;
J=1/(2*m)*sum(sqrerrors);



% =========================================================================

end
