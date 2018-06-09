---
title: 深度学习 QA
author: 张凯
tags: deep-learning
---

本文总结了深度学习面试中常见的问题。

<!--more-->

## CNN最成功的应用是在CV，那为什么NLP和Speech的很多问题也可以用CNN解出来？

<https://zhuanlan.zhihu.com/p/25005808>
  
## 什么样的资料集不适合用深度学习

- 数据集太小。因为神经网络有大量的参数需要训练
- 数据集没有局部相关性。目前深度学习应用的领域图像、语音和自然语言处理，其共性为
  局部相关性。例如，图像中的像素组成物体，语音中音位组成单词，文本中的单词组成句
  子。而深度学习过程就是学习局部低层次的特征，然后组合成高层次的特征，得到不同特
  征之间的空间相关性。

## 对所有优化问题来说, 有没有可能找到比现在已知算法更好的算法？

- 不存在一个通用的模型，对所有的优化问题都能做到性能最佳

![没有免费的午餐](../images/optimise-no-free-lunch.png){#fig:noFreeLunch}

- 一个学习算法 A，若它在某些问题上比学习算法 B 更好，则必然存在一些问题，在那里
  B 比 A 好。也就是说，无论算法 A 多聪明，算法 B 多笨拙，它们的期望性能相等
- 没有免费的午餐定理假设所有问题出现的概率相等。实际应用中，不同的场景，会有不同
  的问题分布。所以算法优化的核心是具体问题具体分析

## 用贝叶斯机率说明 Dropout 的原理

请参考[文献阅读：Dropout as a Bayesian Approximation: Insights and
Applications](./2018-05-29-Dropout-as-a-Bayesian-Approximation-Insights-and-Applications.html)
。

## 何为共线性，跟过拟合有啥关联？

- 共线性指多变量线性回归中，变量之间由于存在高度相关关系而使回归估计不准确。
- 共线性会造成冗余，导致过拟合
- 解决方法：排除变量的相关性/正则化

## 说明如何用支持向量机实现深度学习

## 列举常见的范数及其应用场景，如 $L^0, L^1, L^2, L^\infty$, Frobenius 范数

请参考[文献阅读：Deep Learning#范数](./2018-06-02-deep-learning-goodfellow2016.html#范数)。

## 贝叶斯概率与频率派概率，以及在统计中对于真实参数的假设

- 世界观不同。频率派认为世界是确定的，参数有一个真值，我们的目标是找到这个真值或
  真值所在的范围；贝叶斯派则认为世界是不确定的，参数是随机变量，人们对参数的概率
  分布先有一个预判，而后通过观测数据对这个预判做调整，我们的目标是要找到最优的描
  述参数的概率分布。

## 概率密度的万能近似器

请参考[文献阅读：Deep Learning#混合分布](./2018-06-02-deep-learning-goodfellow2016.html#混合分布)。

## sigmoid，relu，softplus，tanh 和 RBF 及其应用场景

请参考[文献阅读：Deep Learning#常见函数](./2018-06-02-deep-learning-goodfellow2016.html#常见函数)。

## KL 散度

请参考[文献阅读：Deep Learning#信息论](./2018-06-02-deep-learning-goodfellow2016.html#信息论)。

## 数值计算中的计算上溢与下溢问题，如 softmax 中的处理方式

请参考[文献阅读：Deep Learning#上溢和下溢](./2018-06-02-deep-learning-goodfellow2016.html#上溢和下溢)。

## 与矩阵的特征值相关联的条件数（病态条件）指什么，与梯度爆炸和梯度弥散的关系

请参考[文献阅读：Deep Learning#病态条件](./2018-06-02-deep-learning-goodfellow2016.html#病态条件)
和[文献阅读：Deep Learning#长期依赖](./2018-06-02-deep-learning-goodfellow2016.html#长期依赖)。

## 在基于梯度的优化问题中，如何判断一个梯度为 $0$ 的临界点为极大值、极小值还是鞍点？Hessian 矩阵的条件数与梯度下降的关系

请参考[文献阅读：Deep Learning#梯度之外：Jacobian 和 Hessian 矩阵](./2018-06-02-deep-learning-goodfellow2016.html#梯度之外jacobian-和-hessian-矩阵)。

## KTT 方法与约束优化问题，活跃约束的定义

请参考[文献阅读：Deep Learning#有约束的优化](./2018-06-02-deep-learning-goodfellow2016.html#有约束的优化)。

## 模型容量，表示容量，有效容量和最优容量概念

请参考[文献阅读：Deep Learning#容量过拟合和欠拟合](./2018-06-02-deep-learning-goodfellow2016.html#容量过拟合和欠拟合)。

## 正则化中的权重衰减与加入先验知识在某些条件下的等价性

请参考[文献阅读：Deep Learning#Maximum A Posterior (MAP) Estimation（最大后验概率估计）](./2018-06-02-deep-learning-goodfellow2016.html#maximum-a-posteriori-map-estimation最大后验概率估计)。

## 高斯分布的广泛应用的缘由
