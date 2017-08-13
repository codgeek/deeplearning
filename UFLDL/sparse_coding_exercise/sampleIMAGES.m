function patches = sampleIMAGES(patchsize, numpatches)
% sampleIMAGES
% Returns 10000 patches for training

load IMAGES;    % load images from disk 
dataSet=IMAGES;

% addpath('../../mnist');
% dataSet = loadMNISTImages('train-images.idx3-ubyte');
% labels = loadMNISTLabels('train-labels.idx1-ubyte');
% dataSet = dataSet(:,labels > 4);
% % patchsize = 8;  % we'll use 8x8 patches 
% numpatches = (6e4);
% display_network(dataSet(:,1:100))
% dataSet = reshape(dataSet, 28, 28, numPic);
% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches);


%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1
%     figure;
%     subplot(121);imagesc(IMAGES(:,:,6));colormap gray;
%     subplot(122);imagesc(IMAGES(:,:,3));colormap gray;
    numPic = size(dataSet,2);
    [row, colum, numPic] = size(dataSet);
    rowStart = randi(row - patchsize + 1, numpatches, 1);
    columStart = randi(colum - patchsize + 1, numpatches, 1);
    randIdx = randperm(numPic); randIdx = randperm(numPic); % the 3-rd dimension of patches is linear with images index 1:numPic
%     randIdx = 1:numPic;
    for r=1:numpatches
        patches(:,r) = reshape(dataSet(rowStart(r):rowStart(r) + patchsize - 1, columStart(r):columStart(r) + patchsize - 1, randIdx(floor((r-1)*numPic/numpatches)+1)),[],1);
    end

%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
% patches = normalizeData(patches);

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
