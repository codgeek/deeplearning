function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels, ~)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);
% groundTruth = full(sparse(labels, 1:numCases, 1));
% 
labels = repmat(labels, numClasses, 1);
k = repmat((1:numClasses)',1,numCases);% numClasses¡ÁnumCases. 
groundTruth = double((k == labels));% % groundTruth algrithum is the same as (k===label)
thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
cost = 0;
z = theta*data;
z = z - max(max(z)); % avoid overflow while keep p unchanged.
z = exp(z); % matrix product: numClasses¡ÁnumCases
p = z./repmat(sum(z,1),numClasses,1); % normalize the probbility aganist numClasses. numClasses¡ÁnumCases
cost = -mean(sum(groundTruth.*log(p), 1)) + sum(sum(theta.*theta)).*(lambda/2);

thetagrad = -(groundTruth - p)*(data')./numCases + theta.*lambda; % numClasses¡ÁinputSize





% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = thetagrad(:);
end

