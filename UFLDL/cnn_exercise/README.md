# UFLDL深度学习笔记 （六）卷积神经网络



## 1. 主要思路

“[UFLDL **卷积神经网络**](http://deeplearning.stanford.edu/wiki/index.php/%E5%8D%B7%E7%A7%AF%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96)”主要讲解了对大尺寸图像应用前面所讨论神经网络学习的方法，其中的变化有两条，第一，对大尺寸图像的每个小的patch矩阵应用相同的权值来计算隐藏层特征，称为`卷积特征提取`；第二，对计算出来的特征矩阵做“减法”，把特征矩阵纵横等分为多个区域，取每个区域的平均值(或最大值)作为输出特征，称为`池化`。这样做的原因主要是为了降低数据规模，对于8X8的图像输入层有64个单元，而100X100的图像，输入单元有1E4个，相同特征个数下需要训练的权重参数个数呈平方倍增加。真实图像大到一定程度，将会很难运行训练。所以卷积特征提取步骤中，使用在小尺寸patch上获得的权重参数作为`共享的权重`对大尺寸图像每个小patch做卷积运算，来达到前向传播的作用，结果中包含大图像每个小邻域的特征。而池化步骤中对过大的特征矩阵做平均、或取最大值更是明显的“下采样”特征矩阵了。

## 2. 卷积和前向传播的关系

<center><img src="http://images2015.cnblogs.com/blog/1174358/201707/1174358-20170702161455743-1737131402.png" width="500"  /></center>


我们知道对于一个典型的神经网络，训练到的权重参数维度表示为$W^{(1)}_{hiddenSize \times inputSize}  , b^{(1)}_{hiddenSize \times 1} ; _{hiddenSize=patchDim \times patchDim}$,在原文“[UFLDL **卷积神经网络**](http://deeplearning.stanford.edu/wiki/index.php/%E5%8D%B7%E7%A7%AF%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96)”中，所描述的过程是 <u>“取$W^{(1)}$与大图像的每个 $ patchDim \times patchDim $ 子图像做乘法，对 $f_{convoled}$ 值做卷积，就可以得到卷积后的矩阵”</u>  。没有解释清楚经典的前向传播为什么变成了图像卷积的，以及谁和谁卷积。 接下来我们来慢慢理解一下这个转变过程。

对于大图像$X_{r \times c}$我们设想的是取每一个子矩阵$X_{patchDim \times patchDim}$和$W^{(1)}$按照小尺寸图像来做前向传播，所以共需要做$(r-patchDim)\times(c-patchDim)$次前向传播，每一次表示为$h_{hiddenSize \times 1}=\sigma( W^{(1)}*X_{patchDim \times patchDim} (:)+b^{(1)}_{hiddenSize \times 1})$. 所以说会得到$hiddenSize \times (r-patchDim)\times(c-patchDim)$的特征矩阵。接下来需要做一个交换，才能看出来怎么用卷积函数；计算$h_{hiddenSize \times 1}$的步骤里$W^{(1)}$的每一行和和$X_{patchDim \times patchDim}$转成一列向量后进行点积，可以和做$(r-patchDim)\times(c-patchDim)$次前向传播顺序交换，也就是说取$W^{(1)}$的每一行转成$patchDim\times patchDim$的矩阵和$X_{r \times c}$的每一个子矩阵$X_{patchDim \times patchDim}$做点乘求和，**这不正好是二维卷积吗！**，是的，卷积就是这样来的，本质上是对大图像每个小邻域的前向传播，正好运用了二维矩阵卷积这个工具。

## 3. 代码实现

有了前面的认识，我们再来看卷积计算的实现代码, 有两点需要注意的：

- `feature = W(featureNum,:,:,channel);`这一行代表取出$W^{(1)}_{hiddenSize \times inputSize}$的每一行，然后用二维卷积函数对每张图片`im = squeeze(images(:, :, channel, imageNum));`做前向传播`conv2(im, feature, 'valid')`。
- 为什么要对每张图片的RGB三个通道的卷积累加呢? 这是因为训练权重时$W^{(1)}_{hiddenSize \times inputSize}$的$inputSize$是把RGB通道展开变成一维向量后训练出来的，对大图像的小patch做前向传播$W^{(1)}_{hiddenSize \times inputSize}$的每一行与$X_{patchDim \times patchDim}$做点积是RGB三段的累加，所以需要累加三个通道的特征响应，即`convolvedImage = convolvedImage + conv2(im, feature, 'valid');`

池化部分比较直观，不再展开描述，根据`cnnExercise.m`步骤，除了复用原来的稀疏自编码、softmax、栈式自编码部分代码，我们要编写`cnnConvolve.m`，`cnnPool.m`，整体代码详见[https://github.com/codgeek/deeplearning](https://github.com/codgeek/deeplearning/tree/master/UFLDL/cnn_exercise) 

```matlab
function convolvedFeatures = cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  patchDim - patch (feature) dimension
%  numFeatures - number of features
%  images - large images to convolve with, matrix in the form
%           images(r, c, channel, image number)
%  W, b - W, b for features from the sparse autoencoder
%  ZCAWhite, meanPatch - ZCAWhitening and meanPatch matrices used for
%                        preprocessing
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)

numImages = size(images, 4);
imageDim = size(images, 1);
imageChannels = size(images, 3);
% -------------------- YOUR CODE HERE --------------------
% Precompute the matrices that will be used during the convolution. Recall
% that you need to take into account the whitening and mean subtraction
% steps
W = W * ZCAWhite;% W *(ZCAWhite *(X - meanPatch)) equals to (W *ZCAWhite)*X - (W *ZCAWhite)*meanPatch; 
substractMean = W * meanPatch;
W = reshape(W,numFeatures, patchDim, patchDim, imageChannels);

convolvedFeatures = zeros(numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1);
for imageNum = 1:numImages
 for featureNum = 1:numFeatures
    convolvedImage = zeros(imageDim - patchDim + 1, imageDim - patchDim + 1);
    for channel = 1:imageChannels
      % Obtain the feature (patchDim x patchDim) needed during the convolution
      feature = W(featureNum,:,:,channel); % each row of W is one of numFeatures that has learned
      % ------------------------
      % Flip the feature matrix because of the definition of convolution, as explained later
      feature = rot90(squeeze(feature),2);

      % Obtain the image
      im = squeeze(images(:, :, channel, imageNum));
      % Convolve "feature" with "im", adding the result to convolvedImage
      % be sure to do a 'valid' convolution
      % ---- YOUR CODE HERE ----
      convolvedImage = convolvedImage + conv2(im, feature, 'valid');% (imageDim - patchDim + 1) X (imageDim - patchDim + 1)
     % ------------------------
    end
    % Subtract the bias unit (correcting for the mean subtraction as well)
    % Then, apply the sigmoid function to get the hidden activation
    % ---- YOUR CODE HERE ----
    % meanPatch: numFeatures X 1
    convolvedImage = sigmoid(convolvedImage + b(featureNum) - substractMean(featureNum));
    % ------------------------
    % The convolved feature is the sum of the convolved values for all channels
    convolvedFeatures(featureNum, imageNum, :, :) = convolvedImage;
  end
end
end

function sigm = sigmoid(x)
  sigm = 1 ./ (1 + exp(-x));
end

```


## 4.图示与结果

数据集来自[ STL-10 dataset](http://ufldl.stanford.edu/wiki/resources/stl10_patches_100k.zip). 以及我们在前一节[UFLDL深度学习笔记 （五）自编码线性解码器](http://www.cnblogs.com/Deep-Learning/p/7106256.html)中训练得到的该数据集下采样的8X8 patch上的特征参数`STL10Features.mat`，下采样前后图片本身对比如下。


![](http://images2015.cnblogs.com/blog/1174358/201707/1174358-20170709002904175-990789292.png)

![](http://images2015.cnblogs.com/blog/1174358/201707/1174358-20170709002907644-334781897.png)


设定与[练习说明](http://deeplearning.stanford.edu/wiki/index.php/Exercise:Convolution_and_Pooling)相同的参数，输入每个图像为64X64X3的彩色图片，共有四个分类 (airplane, car, cat, dog)。运行代码主文件[cnnExercise.m](https://github.com/codgeek/deeplearning/tree/master/UFLDL/cnn_exercise) 可以看到预测准确率为80.4%。与练习的标准结果吻合。分类的准确其实并不高，一方面原因在于下采样倍率较大，下采样后的图片人眼基本无法分类，而通过特征学习、进一步进行卷积网络学习，仍然可以达到一定的准确率。