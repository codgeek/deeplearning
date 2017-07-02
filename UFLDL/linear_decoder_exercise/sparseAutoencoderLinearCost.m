function [cost,grad] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);


%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% forward propagation

[~, m] = size(data); % visibleSize¡ÁN_samples£¬ m=N_samples
a2 = sigmoid(W1*data + b1*ones(1,m));% active value of hiddenlayer: hiddenSize¡ÁN_samples
a3 = W2*a2 + b2*ones(1,m);% liner decoder would output Z. output result: visibleSize¡ÁN_samples
diff = a3 - data;
penalty = mean(a2, 2); %  measure of hiddenlayer active: hiddenSize¡Á1
residualPenalty =  (-sparsityParam./penalty + (1 - sparsityParam)./(1 - penalty)).*beta; % penalty factor in residual error delta2
% size(residualPenalty)
cost = sum(sum((diff.*diff)))./(2*m) + ...
    (sum(sum(W1.*W1)) + sum(sum(W2.*W2))).*lambda./2 + ...
    beta.*sum(KLdivergence(sparsityParam, penalty));

% back propagation
 delta3 = -(data-a3); % liner decoder: visibleSize¡ÁN_samples
 delta2 = (W2'*delta3 + residualPenalty*ones(1, m)).*(a2.*(1-a2)); %  hiddenSize¡ÁN_samples. !!! => W2'*delta3 not W1'*delta3

 W2grad = (a2*(delta3'))'; % ¨ŒJ(L)=delta(L+1,i)*a(l,j). sum of grade value from N_samples is got by matrix product hiddenSize¡ÁN_samples * N_samples¡ÁvisibleSize. so mean value is caculated by "/N_samples"
 W1grad = (data*(delta2'))';% matrix product  visibleSize¡ÁN_samples * N_samples¡ÁhiddenSize
 
 b1grad = sum(delta2, 2);
 b2grad = sum(delta3, 2);

% mean value across N_sample
W1grad=W1grad./m + lambda.*W1;
W2grad=W2grad./m + lambda.*W2;
b1grad=b1grad./m;
b2grad=b2grad./m;% mean value across N_sample: visibleSize ¡Á1

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function value = KLdivergence(pmean, p)
    value = pmean.*log(pmean./p) + (1- pmean).*log((1 - pmean)./( 1 - p));
end
