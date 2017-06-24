
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# UFLDL深度学习笔记 （一）基本知识与稀疏自编码

## 前言

  近来正在系统研究一下深度学习，作为新入门者，为了更好地理解、交流，准备把学习过程总结记录下来。最开始的规划是先学习理论推导；然后学习一两种开源框架；第三是进阶调优、加速技巧。越往后越要带着工作中的实际问题去做，而不能是空中楼阁式沉迷在理论资料的旧数据中。深度学习领域大牛吴恩达(Andrew Ng)老师的[**UFLDL教程**](http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial) (Unsupervised Feature Learning and Deep Learning)提供了很好的基础理论推导，文辞既系统完备，又明白晓畅，最可贵的是能把在斯坦福大学中的教学推广给全世界，尽心力而广育人，实教授之典范。
    这里记录的学习笔记不是为了重复UFLDL中的推导过程，而是为了一方面补充我没有很快看明白的理论要点，另一方面基于我对课后练习的matlab实现，记录讨论卡壳时间较长的地方。也方便其他学习者，学习过程中UFLDL所有留白的代码模块全是自己编写的，没有查阅网上代码，结果都达到了练习给出的参考标准，并且都是按照矩阵化实现(UFLDL称为矢量化 vectorization)，所以各种matlab实现细节、缘由也会比较清楚，欢迎大家一起讨论共同进步。


## 理论推导中的难点

先上一个经典的自编码神经网络结构图。推导中常用的[**符号表征见于此**](http://deeplearning.stanford.edu/wiki/index.php/%E7%A8%80%E7%96%8F%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E7%AC%A6%E5%8F%B7%E4%B8%80%E8%A7%88%E8%A1%A8)

![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170610123116481-660728486.png)

### 反向传导的前因后果啊
神经网络的最基础的理论是[**反向传导算法**](http://deeplearning.stanford.edu/wiki/index.php/%E5%8F%8D%E5%90%91%E4%BC%A0%E5%AF%BC%E7%AE%97%E6%B3%95)，反向传导的主语是误差代价函数\\(J_{W,b}\\)对网络系数(W，b)的偏导数。那为什么要求偏导呢? 这里就引出了最优化理论，具体的问题一般被建模成许多自变量决定的代价函数，最优化的目标就是找到使代价函数最小化的自变量取值。数学方法告诉我们令偏导为0对应的自变量取值就是最小化代价的点，不幸的是多数实际问题对用偏导为0方程求不出闭合的公式解，神经网络也不例外。为了解决这类问题，可以使用优化领域的梯度下降法，梯度是对每个自变量的偏导组成的向量或矩阵，梯度下降直白地说就是沿着对每个自变量偏导的负方向改变自变量，下降指的是负方向，直到梯度值接近0或代价函数变化很小，即停止查找认为找到了达到最优化的所有自变量数值。 所以为了最小化稀疏自编码误差代价函数，我们需要求出J(W,b)对神经网络各层系数W、b的偏导数组成的梯度矩阵。


在上图的神经网络结构中，输入单个向量\\(x\\),逐层根据参数矩阵W b 计算[**前向传播**](http://deeplearning.stanford.edu/wiki/index.php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)，最后输出向量\(h_{W,b}(x)\), 这样单个向量的代价函数是 $$J_{W,b}=\frac{1}{2}{\left\|(h_{W,b}(x)-y\right\|}^2$$

对所有数据取代价即
![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170610161907762-467948345.png)
接下来在[**反向传导算法**](http://deeplearning.stanford.edu/wiki/index.php/%E5%8F%8D%E5%90%91%E4%BC%A0%E5%AF%BC%E7%AE%97%E6%B3%95)文章中给出了详细的推导过程，推导的结果便是残差、梯度值从输出层到隐藏层的反向传导公式

输出层的残差值   ![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170610162638278-1588776276.png)

残差 值   ![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170610162709528-1358668604.png)


![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170610162431965-1279837877.png)



inlineis \\(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\)

name is $$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$

## exercise代码实现难点

### 训练数据

先看一下训练数据的样子吧

![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170610122850918-1711766400.png)


![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170610105914997-407777552.png)

sampleIMAGES.m模是可以被后续课程复用的，第一次写所以解释的更详细一些。

### 后向传播参数对应、矩阵实现



## 完整代码

	​``` javaScript
	    load IMAGES;    % load images from disk 
	    figure;imagesc(IMAGES(:,:,3));colormap gray;
	    figure;imagesc(IMAGES(:,:,6));colormap gray;
	
	​```



## 卡壳的关键点

## 最终结果![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170610162656840-28643908.png)