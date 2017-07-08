function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);

% pooledFeatures = zeros(numFeatures, numImages, floor(convolvedDim / poolDim), floor(convolvedDim / poolDim));

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim)
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region
%   (see http://ufldl/wiki/index.php/Pooling )
%
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------
numArray = repmat(poolDim,floor(convolvedDim/poolDim), 1);
if mod(convolvedDim,poolDim) ~= 0
    numArray = [numArray; mod(convolvedDim, poolDim)];
end
poolEdge = size(numArray,1);
apart = squeeze(mat2cell(convolvedFeatures,numFeatures, numImages, numArray, numArray));% divid matrix aparts will add one scalar dimension and 2 cell dimesion
cellpool = cellfun(@(p) mean(p,3),cellfun(@(p) mean(p,4),apart,'UniformOutput',false),'UniformOutput',false);
pooledFeatures = zeros(poolEdge,poolEdge,numFeatures, numImages);
for poolRow=1:poolEdge
    for poolCol=1:poolEdge
        pooledFeatures(poolRow,poolCol,:,:) = cell2mat(cellpool(poolRow,poolCol));
    end
end
pooledFeatures = permute(pooledFeatures,[3,4,1,2]);
% for imageCount = 1:numImages
%     for featureCount = 1:numFeatures
%         apart = mat2cell(convolvedFeatures(featureCount, imageCount,:,:), numArray, numArray);
%         pooledFeatures(featureCount, imageCount,:,:) = cellfun(@mean,cellfun(@mean,apart,'UniformOutput',false),'UniformOutput',false);
%     end
% end
end

