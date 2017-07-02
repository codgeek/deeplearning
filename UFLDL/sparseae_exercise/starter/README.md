# UFLDL深度学习笔记 （一）基本知识与稀疏自编码

## 前言

　　近来正在系统研究一下深度学习，作为新入门者，为了更好地理解、交流，准备把学习过程总结记录下来。最开始的规划是先学习理论推导；然后学习一两种开源框架；第三是进阶调优、加速技巧。越往后越要带着工作中的实际问题去做，而不能是空中楼阁式沉迷在理论资料的旧数据中。深度学习领域大牛吴恩达(Andrew Ng)老师的[**UFLDL教程**](http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial) (Unsupervised Feature Learning and Deep Learning)提供了很好的基础理论推导，文辞既系统完备，又明白晓畅，最可贵的是能把在斯坦福大学中的教学推广给全世界，尽心力而广育人，实教授之典范。
　　这里的学习笔记不是为了重复UFLDL中的推导过程，而是为了一方面补充我没有很快看明白的理论要点，另一方面基于我对课后练习的matlab实现，记录讨论卡壳时间较长的地方。也方便其他学习者，学习过程中UFLDL所有留白的代码模块全是自己编写的，没有查网上的代码，经过一番调试结果都达到了练习给出的参考标准，而且都按照矩阵化实现(UFLDL称为矢量化 vectorization)，代码链接见此[https://github.com/codgeek/deeplearning](https://github.com/codgeek/deeplearning)，所以各种matlab实现细节、缘由也会比较清楚，此外从实现每次练习的代码commit中可以清楚看到修改了那些代码，欢迎大家一起讨论共同进步。


## 理论推导中的要点

先上一个经典的自编码神经网络结构图。推导中常用的[**符号表征见于此**](http://deeplearning.stanford.edu/wiki/index.php/%E7%A8%80%E7%96%8F%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E7%AC%A6%E5%8F%B7%E4%B8%80%E8%A7%88%E8%A1%A8)

![](http://images2015.cnblogs.com/blog/1174358/201707/1174358-20170702161455743-1737131402.png)
上图的神经网络结构中，输入单个向量$\ x\ $,逐层根据网络参数矩阵$\ W, b\ $计算[**前向传播**](http://deeplearning.stanford.edu/wiki/index.php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)，最后输出向量$ h_{W,b}(x)$, 这样单个向量的误差代价函数是
$$J_{W,b,x,y}=\frac{1}{2}{||(h_{W,b}(x)-y||}^2$$
对所有数据代价之和即
![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170610161907762-467948345.png)

### 偏导与反向传导的前因后果
神经网络的最基础的理论是[**反向传导算法**](http://deeplearning.stanford.edu/wiki/index.php/%E5%8F%8D%E5%90%91%E4%BC%A0%E5%AF%BC%E7%AE%97%E6%B3%95)，反向传导的主语是误差代价函数$\ J_{W,b}\ $对网络系数(W，b)的偏导数。那为什么要求偏导呢? 这里就引出了最优化理论，具体的问题一般被建模成许多自变量决定的代价函数，最优化的目标就是找到使代价函数最小化的自变量取值。数学方法告诉我们令偏导为0对应的自变量取值就是最小化代价的点，不幸的是多数实际问题对用偏导为0方程求不出闭合的公式解，神经网络也不例外。为了解决这类问题，可以使用优化领域的梯度下降法，梯度是对每个自变量的偏导组成的向量或矩阵，梯度下降直白地说就是沿着对每个自变量偏导的负方向改变自变量，下降指的是负方向，直到梯度值接近0或代价函数变化很小，即停止查找认为找到了达到最优化的所有自变量数值。 所以为了最小化稀疏自编码误差代价函数，我们需要求出J(W,b)对神经网络各层系数W、b的偏导数组成的梯度矩阵。


接下来在[**反向传导算法**](http://deeplearning.stanford.edu/wiki/index.php/%E5%8F%8D%E5%90%91%E4%BC%A0%E5%AF%BC%E7%AE%97%E6%B3%95)文章中给出了详细的推导过程，推导的结果便是残差、梯度值从输出层到隐藏层的反向传导公式

输出层的对输出值$\ z\ $第i个分量$\ z_i^{n_l}\ $的偏导为  ![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170610162638278-1588776276.png)

后向传导时，第l+1层传导给第l层偏导值为   ![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170610162709528-1358668604.png)

### J对输出值z的偏导怎么得到对网络权值参数的偏导？
我们知道$\ z_i^{l+1}\ $与$\ W,b\ $的关系$\ z_i^{(l+1)}=\sum_{j=1}^{n_l} ({W_{i,j}^{(l)}\cdot{a_j^{(l)}+b}})\ $，运用求导链式法则可得：

$$\begin{align} \frac {\partial J_{W,b;x,y}}{\partial W_{i,j}^{(l)}} &= \frac {\partial J_{W,b;x,y}}{\partial z_i^{(l+1)}} \cdot \frac {\partial z_i^{(l+1)}}{\partial W_{i,j}^{(l)}} \\  &=\delta_i^{l+1}\cdot a_j^l\end{align}$$
原文中跳过了这一步，基于此，才能得到误差函数对参数$\ W, b\ $每一项的偏导数。
![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170610162431965-1279837877.png)

### 结合稀疏自编码问题
前面的反向传导是神经网络的一般形式，具体到稀疏自编码，有两个关键字“稀疏”、“自”。“自”是指目标值$\ y\ $和输入向量$\ x\ $相等，“稀疏”是指过程要使隐藏层L1的大部分激活值分量接近0，个别分量显著大于0.所以稀疏自编码的代价函数中$\ y\ $直接使用$\ x\ $的值。同时加上稀疏性惩罚项,详见[稀疏编码一节](http://deeplearning.stanford.edu/wiki/index.php/%E8%87%AA%E7%BC%96%E7%A0%81%E7%AE%97%E6%B3%95%E4%B8%8E%E7%A8%80%E7%96%8F%E6%80%A7)。
$$J_sparse(W,b)=J(W,b)+\beta\sum_{j=1}^{S_2}KL(\rho||\hat\rho_j)$$
惩罚项只施加在隐藏层上，对偏导项做对应项添加，也只有隐藏层的偏导增加一项。
![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170611155007059-154493332.png)
后向传导以及对网络权值$\ W, b\ $的梯度公式不变。

在反向传导的前因后果段落，我们已经知道了要用梯度下降法搜索使代价函数最小的自变量$\ W, b\ $。实际上梯度下降也不是简单减梯度，而是有下降速率参数$\ \alpha\ $的L-BFGS，又牵扯到一个专门的优化。为了简化要做的工作量，[稀疏自编码练习](http://deeplearning.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder)中已经帮我们集成了一个第三方实现，在minFunc文件夹中，我们只需要提供代价函数、梯度计算函数就可以调用minFunc实现梯度下降，得到最优化参数。后续的工作也就是要只需要补上**sampleIMAGES.m, sparseAutoencoderCost.m, computeNumericalGradient.m**的"YOUR CODE HERE"的代码段。

## exercise代码实现难点

UFLDL给大家的学习模式很到家，把周边的结构性代码都写好了matlab代码与注释，尽量给学习者减负。系数自编码中主m文件是train.m。先用实现好的代价、梯度模块调用梯度检验，然后将上述代价、梯度函数传入梯度下降优化minFunc。满足迭代次数后退出，得到训练好的神经网络系数矩阵，补全全部待实现模块的完整代码见此，[https://github.com/codgeek/deeplearning](https://github.com/codgeek/deeplearning).
其中主要过程是
### 训练数据

训练数据来源是图片，第一步要在**sampleIMAGES.m**中将其转化为神经网络$\ x\ $向量。先看一下训练数据的样子吧

![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170611222116528-747739699.png)

sampleIMAGES.m模块是可以被后续课程复用的，尽量写的精简通用些。从一幅图上取patchsize大小的子矩阵，要避免每个图上取的位置相同，也要考虑并行化，循环每张图片取patch效率较低。下文给出的方法是先随机确定行起始行列坐标，计算出numpatches个这样的坐标值，然后就能每个patch不关联地取出来，才能运用parfor做并行循环。

    sampleIMAGES.m
    function patches = sampleIMAGES(patchsize, numpatches)
    	load IMAGES;    % load images from disk dataSet=IMAGES;
        dataSet = IMAGES;
    	patches = zeros(patchsize*patchsize, numpatches); % 初始化数组大小，为并行循环开辟空间
    	[row, colum, numPic] = size(dataSet);
    	rowStart = randi(row - patchsize + 1, numpatches, 1);% 从一幅图上取patchsize大小子矩阵，只需要确定随机行起始点，范围是[1,row - patchsize + 1]
    	columStart = randi(colum - patchsize + 1, numpatches, 1);% 确定随机列起始点，范围是colum - patchsize + 1
    	randIdx = randperm(numPic); % 确定从哪一张图上取子矩阵，打乱排列，防止生成的patch顺序排列。
    	parfor r=1:numpatches % 确定了起始坐标后，每个patch不关联，可以并行循环取
        	patches(:,r) = reshape(dataSet(rowStart(r):rowStart(r) + patchsize - 1, columStart(r):columStart(r) + patchsize - 1, randIdx(floor((r-1)*numPic/numpatches)+1)),[],1);
    	end
    	patches = normalizeData(patches)
    end

### 后向传播模型的矩阵实现
稀疏自编码最重要模块是计算代价、梯度矩阵的sparseAutoencoderCost.m。输入$\ W, b\ $与训练数据data，外加稀疏性因子、稀疏项系数、权重$\ W\ $系数，值得关注的是后向传导时隐藏层输出的残差delta2是由隐藏层与输出层的网络参数$\ W2\ $和输出层的残差delta3决定的，不是输入层与隐藏层的网络参数$\ W1\ $，这里最开始写错了，耽误了一些时间才调试正确。下文结合代码用注释的形式解释每一步具体作用。
    sparseAutoencoderCost.m
    function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)
        %% 一维向量重组为便于矩阵计算的神经网络系数矩阵。
        W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
        W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
        b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
        b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
    
        %% 前向传导，W1矩阵为hiddenSize×visibleSize，训练数据data为visibleSize×N_samples，训练所有数据的过程正好是矩阵相乘$\ W1*data\ $。注意所有训练数据都共用系数$\ b\ $，而单个向量的每个分量对用使用$\ b\ $的对应分量，b1*ones(1,m)是将列向量复制m遍，组成和$\ W1*data\ $相同维度的矩阵。
        [~, m] = size(data); % visibleSize×N_samples， m=N_samples
        a2 = sigmoid(W1*data + b1*ones(1,m));% active value of hiddenlayer: hiddenSize×N_samples
        a3 = sigmoid(W2*a2 + b2*ones(1,m));% output result: visibleSize×N_samples
        diff = a3 - data; % 自编码也就意味着将激活值和原始训练数据做差值。
        penalty = mean(a2, 2); %  measure of hiddenlayer active: hiddenSize×1
        residualPenalty =  (-sparsityParam./penalty + (1 - sparsityParam)./(1 - penalty)).*beta; % penalty factor in residual error delta2
        % size(residualPenalty)
        cost = sum(sum((diff.*diff)))./(2*m) + ...
            (sum(sum(W1.*W1)) + sum(sum(W2.*W2))).*lambda./2 + ...
            beta.*sum(KLdivergence(sparsityParam, penalty));
       
        % 后向传导过程，隐藏层残差需要考虑稀疏性惩罚项，公式比较清晰。
         delta3 = -(data-a3).*(a3.*(1-a3)); %  visibleSize×N_samples
         delta2 = (W2'*delta3 + residualPenalty*ones(1, m)).*(a2.*(1-a2)); %  hiddenSize×N_samples. !!! => W2'*delta3 not W1'*delta3
         % 前面已经推导出代价函数对W2的偏导，矩阵乘法里包含了公式中l层激活值a向量与1+1层残差delta向量的点乘。
         W2grad = delta3*a2'; % ▽J(L)=delta(L+1,i)*a(l,j). sum of grade value from N_samples is got by matrix product visibleSize×N_samples * N_samples× hiddenSize. so mean value is caculated by "/N_samples"
         W1grad = delta2*data';% matrix product  visibleSize×N_samples * N_samples×hiddenSize
         
         b1grad = sum(delta2, 2);
         b2grad = sum(delta3, 2);
    
        % 对m个训练数据取平均
        W1grad=W1grad./m + lambda.*W1;
        W2grad=W2grad./m + lambda.*W2;
        b1grad=b1grad./m;
        b2grad=b2grad./m;% mean value across N_sample: visibleSize ×1
    
        % 矩阵转列向量
        grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
    
    end

## 最终结果
在**tarin.m**中，将sparseAutoencoderCost.m传入梯度优化函数minFunc，经过迭代训练出网络参数，通常见到的各个方向的边缘检测图表示的是权重矩阵对每个隐藏层激活值的特征，当输入值为对应特征值时，每个激活值会有最大响应，同时的其余隐藏节点处于抑制状态。所以只需要把W1矩阵每一行的64个向量还原成8*8的图片patch，也就是特征值了，每个隐藏层对应一个，总共100个。结果如下图。

![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170612002219106-1149699040.png)

接下来我们验证一下隐藏层对应的输出是否满足激活、抑制的要求，将上述输入值用参数矩阵传导到隐藏层，也就是$\ W1*W1'\ $.可见，每个输入对应的隐藏层只有一个像素是白色，表示接近1，其余是暗色调接近0，满足了稀疏性要求，而且激活的像素位置是顺序排列的。
![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170612002224700-1089887054.png)

继续改变不同的输入层单元个数，隐藏层单元个数，可以得到更多有意思的结果！