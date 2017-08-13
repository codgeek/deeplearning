function [cost, grad] = sparseCodingFeatureCost(weightMatrix, featureMatrix, visibleSize, numFeatures, patches, gamma, lambda, epsilon, groupMatrix)
%sparseCodingFeatureCost - given the weights in weightMatrix,
%                          computes the cost and gradient with respect to
%                          the features, given in featureMatrix
% parameters
%   weightMatrix  - the weight matrix. weightMatrix(:, c) is the cth basis
%                   vector.
%   featureMatrix - the feature matrix. featureMatrix(:, c) is the features
%                   for the cth example
%   visibleSize   - number of pixels in the patches
%   numFeatures   - number of features
%   patches       - patches
%   gamma         - weight decay parameter (on weightMatrix)
%   lambda        - L1 sparsity weight (on featureMatrix)
%   epsilon       - L1 sparsity epsilon
%   groupMatrix   - the grouping matrix. groupMatrix(r, :) indicates the
%                   features included in the rth group. groupMatrix(r, c)
%                   is 1 if the cth feature is in the rth group and 0
%                   otherwise.

    if exist('groupMatrix', 'var')
        assert(size(groupMatrix, 2) == numFeatures, 'groupMatrix has bad dimension');
    else
        groupMatrix = eye(numFeatures);
    end

    numExamples = size(patches, 2);

    weightMatrix = reshape(weightMatrix, visibleSize, numFeatures);
    featureMatrix = reshape(featureMatrix, numFeatures, numExamples);

    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   features given in featureMatrix.     
    %   You may wish to write the non-topographic version, ignoring
    %   the grouping matrix groupMatrix first, and extend the 
    %   non-topographic version to the topographic version later.
    % -------------------- YOUR CODE HERE --------------------
    linearError = weightMatrix * featureMatrix - patches;
    normError = sum(sum(linearError .* linearError))./numExamples;
    normWeight = sum(sum(weightMatrix .* weightMatrix));
%     smoothFeature = sqrt(featureMatrix .* featureMatrix + epsilon);
%     prefixParam = repmat(diag(groupMatrix),1,numExamples);
    topoFeature = groupMatrix*(featureMatrix.*featureMatrix);
    smoothFeature = sqrt(topoFeature + epsilon);
    costFeature = sum(sum(smoothFeature));% L1 范数为sum(|x|),对x加上平滑参数后,sum(sqrt(x2+epsilon)).容易错写为sqrt(sum(x2+epsilon))实际是L2范数
    
    cost = normError + lambda.*costFeature + gamma.*normWeight;
    grad = 2./numExamples.*(weightMatrix' * linearError) + lambda.*featureMatrix.*( (groupMatrix')*(1 ./ smoothFeature) );% 不止(f,i)本项偏导非零，(f-1,i)……，groupMatrix第f列不为0的所有行对应项都有s(f,i)的偏导
    grad = grad(:);
end
