%% Machine Learning Online Class
%  Exercise 8 | Anomaly Detection and Collaborative Filtering
%% Initialization
clear ; close all; clc
%% ================== Part 1: Load Example Dataset  ===================
%  We start this exercise by using a small dataset that is easy to  visualize.
%  Our example case consists of 2 network server statistics across
%  several machines: the latency and throughput of each machine.
%  This exercise will help us find possibly faulty (or very fast) machines.
fprintf('Visualizing example dataset for outlier detection.\n\n');

%  You should now have the variables X, Xval, yval in your environment
load('ex8data1.mat');
% 用来训练的数据集假设都是正常的   用交叉验证集去检验 每个Xval对应一个yval
%  Visualize the example dataset
plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Program paused. Press enter to continue.\n');
pause
%% ================== Part 2: Estimate the dataset statistics ===================
%  For this exercise, we assume a Gaussian distribution for the dataset.
%  We first estimate the parameters of our assumed Gaussian distribution, 
%  then compute the probabilities for each of the points and then visualize 
%  both the overall distribution and where each of the points falls in 
%  terms of that distribution.
%
fprintf('Visualizing Gaussian fit.\n\n');
%  Estimate mu and sigma2
[mu sigma2] = estimateGaussian(X); %每一个特征都有对应的高斯分布
%  Returns the density of the multivariate normal at each data point (row) of X
p = multivariateGaussian(X, mu, sigma2); %使用多元高斯分布  sigma 要转化为协方差矩阵
% 上面的P是训练样本所得到的概率模型
%  Visualize the fit
visualizeFit(X,  mu, sigma2);  %可视化概率模型
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================== Part 3: Find Outliers ===================
%  Now you will find a good epsilon threshold using a cross-validation set
%  probabilities given the estimated Gaussian distribution
pval = multivariateGaussian(Xval, mu, sigma2);
%用训练好的概率模型（有mu sigma）去求交叉验证中每个数据的概率值
[epsilon F1] = selectThreshold(yval, pval);% 用交叉验证集选取一个好的阈值 
% 因为数据是倾斜的 所以采用计算F1-sorce来评估 选出使F1-sorce最小的阈值
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 8.99e-05)\n');
fprintf('   (you should see a Best F1 value of  0.875000)\n\n');

%  Find the outliers in the training set and plot the
outliers = find(p < epsilon); % 返回与之对应的线性索引数组X的非零条目。X可以是逻辑表达式。
%  Draw a red circle around those outliers
hold on %将训练集中的异常值找出
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off
fprintf('Program paused. Press enter to continue.\n');
pause;
%% ================== Part 4: Multidimensional Outliers ===================
%  We will now use the code from the previous part and apply it to a 
%  harder problem in which more features describe each datapoint and only 
%  some features indicate whether a point is an outlier.
%  Loads the second dataset. You should now have the
%  variables X, Xval, yval in your environment
load('ex8data2.mat');
%  Apply the same steps to the larger dataset
[mu sigma2] = estimateGaussian(X);

%  Training set 
p = multivariateGaussian(X, mu, sigma2);

%  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2);

%  Find the best threshold
[epsilon F1] = selectThreshold(yval, pval);

fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 1.38e-18)\n');
fprintf('   (you should see a Best F1 value of 0.615385)\n');
fprintf('# Outliers found: %d\n\n', sum(p < epsilon));
