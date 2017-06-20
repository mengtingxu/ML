%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
clear ; close all; clc %初始化
data = load('ex1data2.txt');% Load Data
X = data(:, 1:2);%特征矩阵
y = data(:, 3); 
m = length(y);
%Normalizing Features 
mu = mean(X); % 按列求均值
for i = 1 : m,
	X(i, :) = X(i , :) - mu;% 提取每一行与mu做运算
end
sigma = std(X); %按列求标准差
for i = 1 : m,
	X(i, :) = X(i, :) ./ sigma;
end
X = [ones(m, 1) X];% Add intercept term to X

%gradient descent
alpha = 0.01;% Choose some alpha value
num_iters = 400;
theta = zeros(3, 1);
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
theta = theta - alpha * (X' * ( X * theta - y ) / m);%theta update
predictions =X*theta;  % predictions is hypothesis on all m examples
sqrerrors = (predictions-y).^2;
J_history(iter)=1/(2*m)*sum(sqrerrors);
end

% 画出收敛函数   绘制迭代次数于损失值关系图
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');


fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);% Display gradient descent's result 参数的最终取值

fprintf('Solving with normal equations...\n');
%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

X = [ones(m, 1) X];% Add intercept term to X
% Calculate the parameters from the normal equation
theta = zeros(size(X, 2), 1);
theta = pinv(X' * X) * X' * y;

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

price = 0; % You should change this

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);


