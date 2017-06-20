function [J grad] = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y,lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
%�����������ָ�Ϊ����theta1��theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
      
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));%Ҫ��ÿһ��theta������

% ====================== YOUR CODE HERE ======================

%1.Feedforward the neural network and return the cost in the variable J. 

    a1 = [ones(m,1) X];
    z2 = a1*(Theta1)';
    a2 = [ones(size(z2,1),1) sigmoid(z2)];
    z3 = a2*(Theta2)';
    a3 = sigmoid(z3);
    hx = a3;
y_vec = diag(ones(1,num_labels),0);%��y�����п���ȡֵ������� y����5 ��ɱ�ʾΪy_vec(:,5)��5��
%   y_vec(:,y(i)) ��ʾ��ȡһ������ʱ ������y�Ǽ� ��5��ȡ�����ĵ�5��
for i = 1:m
    J =J + (log(hx(i,:)) * y_vec(:,y(i)) + log(1-hx(i,:)) * (1-y_vec(:,y(i))));
end
J = - J/m;
%regularization with the cost function
theta1=Theta1.^2;
theta2=Theta2.^2;
theta1(:,1)=0;  theta2(:,1)=0; %���򻯵�ʱ��Ҫȥ��theta�еĵ�һ��
rug = lambda/(2*m) * (sum(theta1(:))+sum(theta2(:)));
J = J+rug;

%2.Implement the backpropagation algorithm to compute the gradients  Theta1_grad and Theta2_grad. 
%  that your implementation is correct by running checkNNGradients
del1 = zeros(size(Theta1));
del2 = zeros(size(Theta2));
for i=1:m
    delta3 = (a3(i,:))' - y_vec(:,y(i));
    delta2 = (Theta2)'* delta3 .* sigmoidGradient([1 z2(i,:)]');
    delta2 = delta2(2:end);%��Ϊdelta�а����˺�һ���a0����Ҫ����
    del1 = del1 + delta2*a1(i,:);
    del2 = del2 + delta3*a2(i,:);
    size(del2)
    size(del1)
end
Theta2_grad = del2 / m;%������������
Theta1_grad = del1 / m;

%3.regularization with gradients.
Theta1(:,1)=0; 
Theta2(:,1)=0; %���򻯵�ʱ��Ҫȥ��theta�еĵ�һ��
Theta1_grad = (del1+lambda * Theta1) /m;
Theta2_grad = (del2+lambda * Theta2) /m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
