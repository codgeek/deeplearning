function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels,~)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
numStack = numel(stack);
for d = 1:numStack
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

% forward propagation
activeStack = cell(numStack+1, 1);% first element is input data
activeStack{1} = data;
for d = 2:numStack+1
    activeStack{d} = sigmoid((stack{d-1}.w)*activeStack{d-1} + repmat(stack{d-1}.b, 1, M));
end

z = softmaxTheta*activeStack{numStack+1};%  softmaxTheta：numClasse×hiddenSize. Z:numClasses×numCases
z = z - max(max(z)); % avoid overflow while keep p unchanged.
za = exp(z); % matrix product: numClasses×numCases
p = za./repmat(sum(za,1),numClasses,1); % normalize the probbility aganist numClasses. numClasses×numCases
cost = -mean(sum(groundTruth.*log(p), 1)) + sum(sum(softmaxTheta.*softmaxTheta)).*(lambda/2);


% back propagation
softmaxThetaGrad = -(groundTruth - p)*(activeStack{numStack+1}')./M + softmaxTheta.*lambda; % numClasses×inputSize
% a=exp(z);
% lastLayerDelta = -((1./a).*(groundTruth - p));% .*(z.*(1-z));% ▽J/▽a = softmaxTheta'*(groundTruth - p) :numClasses×numCases . Z:numClasses×numCases %res of softmax output layer 
lastLayerDelta = -(groundTruth - p);%各层残差delta定义是J对各层z的偏导数，不是激活值a, 输出层残差delta是▽J/▽z，没有1/a(i,j) 这个系数
lastLayerDelta = (softmaxTheta')*lastLayerDelta.*(activeStack{numStack+1}.*(1-activeStack{numStack+1})); % res of softmax input layer
for d = numel(stack):-1:1
    stackgrad{d}.w = (activeStack{d}*lastLayerDelta')'./M;
    stackgrad{d}.b = mean(lastLayerDelta, 2);
    lastLayerDelta = ((stack{d}.w)')*lastLayerDelta.*(activeStack{d}.*(1-activeStack{d}));
end

%  delta3 = -(data-a3).*(a3.*(1-a3)); %  visibleSize×N_samples
% %  delta2 = (W2'*delta3 ).*(a2.*(1-a2)); %  hiddenSize×N_samples. !!! => W2'*delta3 not W1'*delta3
%  delta2 = (W2'*delta3 + residualPenalty*ones(1, m)).*(a2.*(1-a2)); %  hiddenSize×N_samples. !!! => W2'*delta3 not W1'*delta3
% 
%  W2grad = (a2*(delta3'))'; % ▽J(L)=delta(L+1,i)*a(l,j). sum of grade value from N_samples is got by matrix product hiddenSize×N_samples * N_samples×visibleSize. so mean value is caculated by "/N_samples"
%  W1grad = (data*(delta2'))';% matrix product  visibleSize×N_samples * N_samples×hiddenSize
%  
%  b1grad = sum(delta2, 2);
%  b2grad = sum(delta3, 2);
% 
% % mean value across N_sample
% W1grad=W1grad./m + lambda.*W1;
% W2grad=W2grad./m + lambda.*W2;
% b1grad=b1grad./m;
% b2grad=b2grad./m;% mean value across N_sample: visibleSize ×1







% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
