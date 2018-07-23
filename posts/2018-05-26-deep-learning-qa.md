---
title: 深度学习 QA
author: 张凯
tags: deep-learning
---

本文总结了深度学习面试中常见的问题。

<!--more-->

1. CNN最成功的应用是在CV, 那为什么NLP和Speech的很多问题也可以用CNN解出来？

   <https://zhuanlan.zhihu.com/p/25005808>
  
1. 什么样的资料集不适合用深度学习

   - 数据集太小。因为神经网络有大量的参数需要训练
   - 数据集没有局部相关性。目前深度学习应用的领域图像、语音和自然语言处理，其共性为
     局部相关性。例如，图像中的像素组成物体，语音中音位组成单词，文本中的单词组成句
     子。而深度学习过程就是学习局部低层次的特征，然后组合成高层次的特征，得到不同特
     征之间的空间相关性。

1. 对所有优化问题来说, 有没有可能找到比现在已知算法更好的算法？

   - 不存在一个通用的模型，对所有的优化问题都能做到性能最佳

   ![没有免费的午餐](../images/optimise-no-free-lunch.png){#fig:noFreeLunch}

   - 一个学习算法 A，若它在某些问题上比学习算法 B 更好，则必然存在一些问题，在那里
     B 比 A 好。也就是说，无论算法 A 多聪明，算法 B 多笨拙，它们的期望性能相等
   - 没有免费的午餐定理假设所有问题出现的概率相等。实际应用中，不同的场景，会有不同
     的问题分布。所以算法优化的核心是具体问题具体分析

1. 用贝叶斯机率说明 Dropout 的原理

   请参考[文献阅读：Dropout as a Bayesian Approximation: Insights and
   Applications](./2018-05-29-Dropout-as-a-Bayesian-Approximation-Insights-and-Applications.html)。

1. 何为共线性，跟过拟合有啥关联？

   - 共线性指多变量线性回归中，变量之间由于存在高度相关关系而使回归估计不准确。
   - 共线性会造成冗余，导致过拟合
   - 解决方法：排除变量的相关性/正则化

1. 说明如何用支持向量机实现深度学习

## 《Deep Learning》

1. 列举常见的范数及其应用场景，如 $L^0, L^1, L^2, L^\infty$, Frobenius 范数

   请参考[文献阅读：Deep Learning#范数](./2018-06-02-deep-learning-goodfellow2016.html#范数)。

1. 贝叶斯概率与频率派概率，以及在统计中对于真实参数的假设

   世界观不同。频率派认为世界是确定的，参数有一个真值，我们的目标是找到这个真值或
      真值所在的范围；贝叶斯派则认为世界是不确定的，参数是随机变量，人们对参数的概率
      分布先有一个预判，而后通过观测数据对这个预判做调整，我们的目标是要找到最优的描
      述参数的概率分布。

1. 概率密度的万能近似器

   请参考[文献阅读：Deep Learning#混合分布](./2018-06-02-deep-learning-goodfellow2016.html#混合分布)。

1. sigmoid, relu, softplus, tanh 和 RBF 及其应用场景

   请参考[文献阅读：Deep Learning#常见函数](./2018-06-02-deep-learning-goodfellow2016.html#常见函数)。

1. KL 散度

   请参考[文献阅读：Deep Learning#信息论](./2018-06-02-deep-learning-goodfellow2016.html#信息论)。

1. 数值计算中的计算上溢与下溢问题，如 softmax 中的处理方式

   请参考[文献阅读：Deep Learning#上溢和下溢](./2018-06-02-deep-learning-goodfellow2016.html#上溢和下溢)。

1. 与矩阵的特征值相关联的条件数（病态条件）指什么，与梯度爆炸和梯度弥散的关系

   请参考[文献阅读：Deep Learning#病态条件](./2018-06-02-deep-learning-goodfellow2016.html#病态条件)
和[文献阅读：Deep Learning#长期依赖](./2018-06-02-deep-learning-goodfellow2016.html#长期依赖)。

1. 在基于梯度的优化问题中，如何判断一个梯度为 $0$ 的临界点为极大值、极小值还是鞍点？Hessian 矩阵的条件数与梯度下降的关系

   请参考[文献阅读：Deep Learning#梯度之外：Jacobian 和 Hessian 矩阵](./2018-06-02-deep-learning-goodfellow2016.html#梯度之外jacobian-和-hessian-矩阵)。

1. KTT 方法与约束优化问题，活跃约束的定义

   请参考[文献阅读：Deep Learning#有约束的优化](./2018-06-02-deep-learning-goodfellow2016.html#有约束的优化)。

1. 模型容量，表示容量，有效容量和最优容量概念

   请参考[文献阅读：Deep Learning#容量过拟合和欠拟合](./2018-06-02-deep-learning-goodfellow2016.html#容量过拟合和欠拟合)。

1. 正则化中的权重衰减与加入先验知识在某些条件下的等价性

   请参考[文献阅读：Deep Learning#Maximum A Posterior (MAP) Estimation（最大后验概率估计）](./2018-06-02-deep-learning-goodfellow2016.html#maximum-a-posteriori-map-estimation最大后验概率估计)。

1. 高斯分布的广泛应用的缘由

   请参考[文献阅读：Deep Learning#Gaussian Distribution](./2018-06-02-deep-learning-goodfellow2016.html#gaussian-distribution)。

1. 稀疏表示，低维表示和独立表示

   请参考[文献阅读：Deep Learning#非监督学习算法](./2018-06-02-deep-learning-goodfellow2016.html#非监督学习算法)。

1. 列举一些无法基于梯度优化的代价函数及其具有的特点

   请参考[文献阅读：Deep Learning#构建一个机器学习算法](./2018-06-02-deep-learning-goodfellow2016.html#构建一个机器学习算法)。

1. 梯度爆炸的一些解决办法

   请参考[文献阅读：Deep Learning#参数初始化策略](./2018-06-02-deep-learning-goodfellow2016.html#参数初始化策略)。

1. 万能近似定理

   请参考[文献阅读：Deep Learning#架构设计](./2018-06-02-deep-learning-goodfellow2016.html#架构设计)。

1. 在前馈网络中，深度与宽度的关系及表示能力的差异

   请参考[文献阅读：Deep Learning#架构设计](./2018-06-02-deep-learning-goodfellow2016.html#架构设计)。

1. 为什么交叉损失熵损失可以提高具有 Sigmoid 和 Softmax 输出的模型的性能，而使
   用均方误差损失则会存在很多问题？分段线性隐藏层代替 Sigmoid 的利弊。

   请参考[文献阅读：Deep Learning#历史说明](./2018-06-02-deep-learning-goodfellow2016.html#历史说明)。

1. 表示学习发展的初衷？并介绍其典型例子：自编码器

   请参考[文献阅读：Deep Learning#介绍](./2018-06-02-deep-learning-goodfellow2016.html#介绍)。

1. 在做正则化过程中，为什么只对权重做正则惩罚，而不对偏置做权重惩罚？

   请参考[文献阅读：Deep Learning#参数范数惩罚](./2018-06-02-deep-learning-goodfellow2016.html#参数范数惩罚)。

1. 在深度学习神经网络中，所有的层中使用相同权重的利弊

   请参考[文献阅读：Deep Learning#参数范数惩罚](./2018-06-02-deep-learning-goodfellow2016.html#参数范数惩罚)。

1. 正则化过程中，权重衰减与 Hessian 矩阵中特征值的关系

   请参考[文献阅读：Deep Learning#参数范数惩罚](./2018-06-02-deep-learning-goodfellow2016.html#参数范数惩罚)。

1. $L^1$/$L^2$ 正则化与高斯先验/对数先验的 MAP 贝叶斯推断的关系

   请参考[文献阅读：Deep Learning#参数范数惩罚](./2018-06-02-deep-learning-goodfellow2016.html#参数范数惩罚)。

1. 什么是欠约束？为什么大多数的正则化可以使欠约束下的欠定问题在迭代过程中收敛？

   请参考[文献阅读：Deep Learning#正则化和欠约束问题](./2018-06-02-deep-learning-goodfellow2016.html#正则化和欠约束问题)。

1. 为什么考虑在模型训练时对输入（隐藏单元/权重）添加方差较小的噪声，与正则化的关系

   请参考[文献阅读：Deep Learning#数据集扩充](./2018-06-02-deep-learning-goodfellow2016.html#数据集扩充)
   和[文献阅读：Deep Learning#噪声健壮性](./2018-06-02-deep-learning-goodfellow2016.html#噪声健壮性)。

1. 共享参数的概念及在深度学习中的广泛影响

   请参考[文献阅读：Deep Learning#多任务学习](./2018-06-02-deep-learning-goodfellow2016.html#多任务学习)
   和[文献阅读：Deep Learning#参数类型和参数共享](./2018-06-02-deep-learning-goodfellow2016.html#参数类型和参数共享)。

1. Dropout 与 Bagging 集成方法的关系，以及 Dropout 带来的意义与其强大的原因

   请参考[文献阅读：Deep Learning#dropout](./2018-06-02-deep-learning-goodfellow2016.html#dropout)。

1. 批量梯度下降法更新过程中，批量的大小与各种更新的稳定性关系

   请参考[文献阅读：Deep Learning#学习与纯优化有何不同](./2018-06-02-deep-learning-goodfellow2016.html#学习与纯优化有何不同)。

1. 如何避免深度学习中的病态、鞍点、梯度爆炸和梯度弥散？

   请参考[文献阅读：Deep Learning#神经网络优化中的挑战](./2018-06-02-deep-learning-goodfellow2016.html#神经网络优化中的挑战)。

1. SGD 预计学习率的选择方法，带动量的 SGD 对于 Hessian 矩阵病态条件数及随机梯度方差的影响

   请参考[文献阅读：Deep Learning#基本算法](./2018-06-02-deep-learning-goodfellow2016.html#基本算法)。

1. 初始化权重过程中，权重大小在各种网络结构中的影响，以及一些初始化的方法；偏置的初始化

   请参考[文献阅读：Deep Learning#参数初始化策略](./2018-06-02-deep-learning-goodfellow2016.html#参数初始化策略)。

1. 自适应学习率算法 AdaGrad、RMSProp 和 Adam 等算法的做法

   请参考[文献阅读：Deep Learning#自适应学习率算法](./2018-06-02-deep-learning-goodfellow2016.html#自适应学习率算法)。

1. 二阶近似方法牛顿法、共轭梯度和 BGFS 等的做法

   请参考[文献阅读：Deep Learning#二阶近似方法](./2018-06-02-deep-learning-goodfellow2016.html#二阶近似方法)。

1. Hessian 矩阵的标准化对于高阶优化算法的意义。

   请参考[文献阅读：Deep Learning#优化策略和元算法](./2018-06-02-deep-learning-goodfellow2016.html#优化策略和元算法)。

1. 卷积网络中的平移等变性的原因，常见的一些卷积形式

   请参考[文献阅读：Deep Learning#基本卷积函数的变体](./2018-06-02-deep-learning-goodfellow2016.html#基本卷积函数的变体)。
