function [X_norm, mu, sigma] = featureNormalize(X)
m = size(X , 1);
mu = mean(X); % 按列求均值
for i = 1 : m,
	X_norm(i, :) = X(i , :) - mu;% 提取每一行与mu做运算
end
sigma = std(X); %按列求标准差
for i = 1 : m,
	X_norm(i, :) = X_norm(i, :) ./ sigma;
end

end
