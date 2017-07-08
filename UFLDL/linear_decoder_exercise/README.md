# UFLDL深度学习笔记 （五）自编码线性解码器


## 1. 基本问题

在第一篇 [UFLDL深度学习笔记 （一）基本知识与稀疏自编码](http://www.cnblogs.com/Deep-Learning/p/6978115.html)中讨论了激活函数为$sigmoid$函数的系数自编码网络，本文要讨论“[UFLDL **线性解码器**](http://deeplearning.stanford.edu/wiki/index.php/%E7%BA%BF%E6%80%A7%E8%A7%A3%E7%A0%81%E5%99%A8)”，区别在于输出层去掉了$sigmoid$，将计算值$z$直接作为输出。线性输出的原因是为了避免对输入范围的缩放：

*S 型激励函数输出范围是 [0,1]，当$ f(z^{(3)}) $采用该激励函数时，就要对输入限制或缩放，使其位于 [0,1] 范围中。一些数据集，比如 MNIST，能方便将输出缩放到 [0,1] 中，但是很难满足对输入值的要求。比如， PCA 白化处理的输入并不满足 [0,1] 范围要求，也不清楚是否有最好的办法可以将数据缩放到特定范围中。*

<center><img src="http://images2015.cnblogs.com/blog/1174358/201707/1174358-20170702173839258-237413820.png" width="500"  /></center>

既然改变了输出层激活函数，可以想到需要对其残差、偏导公式关系重新推演。

## 2. 公式推导

线性输出的神经网络仍然是三层，$n_l=3$，自编码线性输出$a_i^{(n_l)}$，则$f'(z_i^{(n_l)})=1$,计算输出层残差：

$$\begin{align}        \delta_i^{(3)}  &= -(y_i-a_i^{(n_l)})*f'(z_i^{(n_l)}) \\  &= -(y_i-a_i^{(n_l)})  \\         \end{align}$$

使用反向传播计算另外两层残差：

$$ \begin{align}        \delta^{(2)}  &= {W^{(2)}}^T*\delta^{(3)} .* f'(z_i^{(2)}) \\                 &= {W^{(2)}}^T*\delta^{(3)} .*(a^{(2)}.*(1-a^{(2)}))  \end{align} $$

根据梯度与残差矩阵的关系可得：

$$\begin{align}         \frac {\nabla J} {\nabla W^{(2)}}  & =\frac 1 m \delta^{(3)}*a^{(2)}  \\    \frac {\nabla J} {\nabla b^{(2)}}     &=\frac 1 m\delta^{(3)}          \end{align}   $$ 



同理可求出：

$$ \begin{align}        \delta^{(1)}  &= {W^{(1)}}^T*\delta^{(2)} .* f'(z_i^{(1)}) \\                 &= {W^{(1)}}^T*\delta^{(2)} .*(a^{(1)}.*(1-a^{(1)}))  \end{align} $$

$$\begin{align}         \frac {\nabla J} {\nabla W^{(1)}}  & = \frac 1 m\delta^{(2)}*a^{(1)}  \\    \frac {\nabla J} {\nabla b^{(1)}}     &=\frac 1 m\delta^{(2)}          \end{align}   $$ 

这样就得到了线性解码器自编码网络代价函数对网络权值$W^{(1)}, b^{(1)}; W^{(2)}, b^{(2)}$的梯度。



## 3. 代码实现

根据前面的步骤描述，与稀疏自编码的区别仅仅是梯度公式形式的差异，基本流程以及惩罚项、稀疏性约束完全复用稀疏自编码的要求。需要增加的模块是代价函数与梯度计算模块`sparseAutoencoderLinearCost.m`，详见[https://github.com/codgeek/deeplearning](https://github.com/codgeek/deeplearning/tree/master/UFLDL/stackedae_exercise/linear_decoder_exercise) 

```matlab
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
% forward propagation

[~, m] = size(data); % visibleSize×N_samples， m=N_samples
a2 = sigmoid(W1*data + b1*ones(1,m));% active value of hiddenlayer: hiddenSize×N_samples
a3 = W2*a2 + b2*ones(1,m);% liner decoder would output Z. output result: visibleSize×N_samples
diff = a3 - data;
penalty = mean(a2, 2); %  measure of hiddenlayer active: hiddenSize×1
residualPenalty =  (-sparsityParam./penalty + (1 - sparsityParam)./(1 - penalty)).*beta; % penalty factor in residual error delta2
% size(residualPenalty)
cost = sum(sum((diff.*diff)))./(2*m) + ...
    (sum(sum(W1.*W1)) + sum(sum(W2.*W2))).*lambda./2 + ...
    beta.*sum(KLdivergence(sparsityParam, penalty));

% back propagation
 delta3 = -(data-a3); % liner decoder: visibleSize×N_samples
 delta2 = (W2'*delta3 + residualPenalty*ones(1, m)).*(a2.*(1-a2)); %  hiddenSize×N_samples. !!! => W2'*delta3 not W1'*delta3

 W2grad = (a2*(delta3'))'; % ▽J(L)=delta(L+1,i)*a(l,j). sum of grade value from N_samples is got by matrix product hiddenSize×N_samples * N_samples×visibleSize. so mean value is caculated by "/N_samples"
 W1grad = (data*(delta2'))';% matrix product  visibleSize×N_samples * N_samples×hiddenSize
 
 b1grad = sum(delta2, 2);
 b2grad = sum(delta3, 2);

% mean value across N_sample
W1grad=W1grad./m + lambda.*W1;
W2grad=W2grad./m + lambda.*W2;
b1grad=b1grad./m;
b2grad=b2grad./m;% mean value across N_sample: visibleSize ×1
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function value = KLdivergence(pmean, p)
    value = pmean.*log(pmean./p) + (1- pmean).*log((1 - pmean)./( 1 - p));
end

```

## 4.图示与结果

数据集来自[ STL-10 dataset](http://ufldl.stanford.edu/wiki/resources/stl10_patches_100k.zip). 需要注意的是我们使用的是下采样之后的图片，每张图片为8X8的彩色图片；另外也原始数据需要做ZCA白化处理，得益于matlab丰富的库函数，svd分解、白化等每个步骤只需要单行代码即可完成。

```matlab
% Apply ZCA whitening
sigma = patches * patches' / numPatches;
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
patches = ZCAWhite * patches;
```
![](http://images2015.cnblogs.com/blog/1174358/201707/1174358-20170702160615071-374120802.png)
STL-10 原始图片下采样到8X8像素图片


![](http://images2015.cnblogs.com/blog/1174358/201707/1174358-20170702160627868-1012818722.png)
设定与[练习说明](http://deeplearning.stanford.edu/wiki/index.php/Exercise:Learning_color_features_with_Sparse_Autoencoders)相同的参数，STL10数据为8X8像素的彩色图片，所以输入层是192个单元，隐藏层设定400个节点，输出层同样是192个节点。运行代码主文件[linearDecoderExercise.m](https://github.com/codgeek/deeplearning/tree/master/UFLDL/linear_decoder_exercise) 可以学习到彩色图片特征，如上图所示，本节只是将数据提取为特征，并不进行进一步分类，特征数据留给后续的卷积神经网络使用。