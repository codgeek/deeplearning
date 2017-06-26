# UFLDL深度学习笔记 （三）无监督特征学习



## 1. 主题思路

“[UFLDL 无监督特征学习](http://deeplearning.stanford.edu/wiki/index.php/%E8%87%AA%E6%88%91%E5%AD%A6%E4%B9%A0)”本节全称为**自我学习与无监督特征学习**，和前一节[softmax回归](http://www.cnblogs.com/Deep-Learning/p/7073744.html)很类似，所以本篇笔记会比较简化，主题思路和步骤如下：

- 是把有标签数据分为两份，先对一份原始数据做[无监督的稀疏自编码](http://www.cnblogs.com/Deep-Learning/p/6978115.html)训练，获得输入层到隐藏层的最优化权值参数$W, b$；
- 把另一份数据分成分成训练集与测试集，都送入该参数对应的第一层网络(去掉输出层的稀疏自编码网络)；
- 用训练集输出的特征作为输入，训练softmax分类器；
- 再用此参数对测试集进行分类，计算出分类准确率。

后面两个步骤和前一节[softmax回归](http://www.cnblogs.com/Deep-Learning/p/7073744.html)相同，区别仅仅是输入变成了原始数据的稀疏自编码隐藏层激活值。

![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170627011055852-1849729455.png)

特征提取网络组合形态

## 2. 本节练习不需要公式推导

## 3. 代码实现

根据前面的步骤描述，我们要实现对有监督数据的特征提取，为了报出稀疏自编码模块`sparseAutoencoderCost.m`的独立性，单独写一个`feedForwardAutoencoder.m`模块来实现。代码是不是非常简单？

```matlab
function [activation] = feedForwardAutoencoder(theta, hiddenSize, visibleSize, data)
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
m = size(data,2);
activation = sigmoid(W1*data + repmat(b1,1,m));
end
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
```

从UFLDL原有代码中可以发现分类标签从0-9移动到了1-10，原因在于matlab的数组索引从1开始，最后做softmax判决时，找最大概率的分类的类别号得到的是1-10，所以才做移动的。

此外[练习说明](http://deeplearning.stanford.edu/wiki/index.php/Exercise:Self-Taught_Learning)中将类别为6-10作为无监督学习的素材，1-5作为有监督softmax回归的素材，无非就是为了判决出类别号后，统计准确率时可以将`softmaxPredict.m`输出的类别号与测试集的真实label直接比较，不用考虑偏移的问题。

## 4.图示与结果

数据集仍然来自[Yann Lecun的笔迹数据库](http://yann.lecun.com/exdb/mnist/)，再瞜一眼原始MMIST数据集的笔迹。

![handwritting](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170625214818132-1942610896.png)



设定与[练习说明](http://deeplearning.stanford.edu/wiki/index.php/Exercise:Softmax_Regression)相同的参数，运行完整代码[https://github.com/codgeek/deeplearning](https://github.com/codgeek/deeplearning) 可以看到预测准确率达到98.35%。达到了练习的标准结果。

![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170627011134368-263195552.png)

回过来看一下，无监督学习到底学习到了什么呢？

![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170627011149086-1903479646.png)

类似稀疏自编码对边缘的学习，上图的特征其实还是稀疏自编码学习到的阿拉伯数字的笔迹基本特征，各种弧线样式。另外一点有点讲究的是对总数据集合拆分两组，也可以不按照1-5、6-10来拆，如果按照数据集的前一半后一段拆成两组。这样无监督训练数据就是所有0-9的数字笔迹，softmax判决的也是0-9的数字笔迹。代码见`stlExercise.m`的`% 拆分无监督学习和softmax监督学习方法二 `部分。

![](http://images2015.cnblogs.com/blog/1174358/201706/1174358-20170627011156602-1171477272.png)

结果有些意外又可以解释：准确率为95.8%。 可以解释为稀疏自编码、softmax有监督学习对每个分类的数据量都减少了，在1E5量级上，数据量对结果好坏有明显的影响。

