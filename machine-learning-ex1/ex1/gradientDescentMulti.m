function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
theta = theta - alpha * (X' * ( X * theta - y ) / m);%theta update
predictions =X*theta;  % predictions is hypothesis on all m examples
sqrerrors = (predictions-y).^2;
J_history(iter)=1/(2*m)*sum(sqrerrors);
end

end
