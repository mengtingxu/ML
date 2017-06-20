function plotData(X, y)
figure; hold on;
pos = find(y==1); neg = find(y == 0);% Find Indices of Positive and Negative Examples 
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);  %行数为pos中的值
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
hold off;

end
