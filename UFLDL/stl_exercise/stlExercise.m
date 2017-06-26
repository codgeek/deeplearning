%% CS294A/CS294W Self-taught Learning Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises.
%
%% ======================================================================
%  STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize  = 28 * 28;% 28*28
numLabels  = 5;      % 5
hiddenSize = 196;    % 200
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 3e-3;       % weight decay parameter       3e-3 
beta = 3;            % weight of sparsity penalty term   3
maxIter = 400;

%% ======================================================================
%  STEP 1: Load data from the MNIST database
%
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.
addpath('../mnist');
addpath('../sparseae_exercise/starter/minFunc');
addpath('../sparseae_exercise/starter');
addpath('../softmax_exercise');
% Load MNIST database files
mnistData   = loadMNISTImages('train-images.idx3-ubyte');
mnistLabels = loadMNISTLabels('train-labels.idx1-ubyte');

% 拆分无监督学习和softmax监督学习方法一
% % 由于流程是在6-10上无监督学习，1-5上有监督学习，如果把0映射到10，识别率稳定在97.3%， 而如果把label号+1平移，识别率稳定在98.3%， 数据不够均匀引起的
% mnistLabels = mnistLabels + 1;% shift 0-9 to 1-10 %
% unlabeledData = mnistData(:,mnistLabels>5);
% labeledData = mnistData(:,mnistLabels<6);
% labels = mnistLabels(mnistLabels<6);

% 拆分无监督学习和softmax监督学习方法二 
% % 对数据无监督学习和监督学习的处理如果不区分0-4还是5-9，效果反而更差，为什么？ 
% % ！！ans: 关键参数有遗漏, 不区分区间，决策范围是1-10.numLabels = 10;
len = size(mnistData, 2);
numLabels = 10;
mnistLabels = mnistLabels + 1;% shift 0-9 to 1-10
unlabeledData = mnistData(:,1:len/2);
labeledData = mnistData(:,len/2 + 1:end);
labels = mnistLabels(len/2 + 1:end);


len = size(labeledData, 2);
trainLen = bitshift(len, -1);
trainData   = labeledData(:, 1:trainLen);
trainLabels = labels( 1:trainLen)';

testData = labeledData(:, trainLen+1:end);
testLabels = labels(trainLen+1:end)';



% Output Some Statistics
fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in supervised training set: %d\n\n', size(trainData, 2));
fprintf('# examples in supervised testing set: %d\n\n', size(testData, 2));

%% ======================================================================
%  STEP 2: Train the sparse autoencoder
%  This trains the sparse autoencoder on the unlabeled training
%  images. 

%  Randomly initialize the parameters


%% ----------------- YOUR CODE HERE ----------------------
%  Find opttheta by running the sparse autoencoder on
%  unlabeledTrainingImages
DEBUG = false;
if DEBUG
    theta = initializeParameters(hiddenSize, inputSize);
    tic
    [cost, grad] = sparseAutoencoderCost(theta, inputSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, unlabeledData(:,1:100), false);
    numgrad = computeNumericalGradient( @(x, true) sparseAutoencoderCost(x, inputSize, ...
                                                      hiddenSize, lambda, ...
                                                      sparsityParam, beta, ...
                                                      unlabeledData(:,1:100), true), theta);

    toc
    % Compare numerically computed gradients with the ones obtained from backpropagation
    diff = norm(numgrad-grad)/norm(numgrad+grad);
    disp(diff); % Should be small. In our implementation, these values are usually less than 1e-9.
    return;
end
theta = initializeParameters(hiddenSize, inputSize);
options.Method = 'lbfgs';
options.maxIter = maxIter;
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, unlabeledData), ...
                              theta, options);


%% -----------------------------------------------------
                          
% Visualize weights
W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
display_network(W1');

%%======================================================================
%% STEP 3: Extract Features from the Supervised Dataset
%  
%  You need to complete the code in feedForwardAutoencoder.m so that the 
%  following command will extract features from the data.

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       trainData);

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData);

%%======================================================================
%% STEP 4: Train the softmax classifier

softmaxModel = struct;  
%% ----------------- YOUR CODE HERE ----------------------
%  Use softmaxTrain.m from the previous exercise to train a multi-class
%  classifier. 

%  Use lambda = 1e-4 for the weight regularization for softmax
lambda = 1e-4;
% You need to compute softmaxModel using softmaxTrain on trainFeatures and
% trainLabels

options.maxIter = maxIter;
softmaxModel = softmaxTrain(hiddenSize, numLabels, lambda, ...
                            trainFeatures, trainLabels, options);






%% -----------------------------------------------------


%%======================================================================
%% STEP 5: Testing 

%% ----------------- YOUR CODE HERE ----------------------
% Compute Predictions on the test set (testFeatures) using softmaxPredict
% and softmaxModel

[pred] = softmaxPredict(softmaxModel, testFeatures);






%% -----------------------------------------------------

% Classification Score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

% (note that we shift the labels by 1, so that digit 0 now corresponds to
%  label 1)
%
% Accuracy is the proportion of correctly classified images
% The results for our implementation was:
%
% Accuracy: 98.3%
%
% 
