function [ sae1Features ] = feedForwardAutoencoder( sae1OptTheta, hiddenSizeL1, inputSize, trainData )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

W1 = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize),hiddenSizeL1, inputSize);
b1 = sae1OptTheta(2*hiddenSizeL1*inputSize+1 : 2*hiddenSizeL1*inputSize+hiddenSizeL1);
sae1Features = sigmoid(W1*trainData + repmat(b1,1,size(trainData,2)));

end

function fx = sigmoid(x)
fx = 1./(1 + exp(-x));
end
