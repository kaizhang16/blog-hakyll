---
title: 决策树介绍
author: 张凯
tags: machine-learning
---

本文介绍了决策树。

<!--more-->

## 分类树

### ID3

信息增益
: $$g(D, A) = H(D) - H(D|A)$$
其中，
$$H(D) = -\sum_{k=1}^K \frac{|C_k|}{|D|}\log_2\frac{|C_k|}{|D|}$$
$$H(D|A) = -\sum_{i=1}^n \frac{|D_i|}{|D|}H(D_i)$$

### C4.5

以信息增益作为划分训练数据集的特征，会倾向于选择取值较多的特征。

信息增益比
: $$g_R(D, A) = \frac{g(D, A)}{H_A(D)}$$
其中，
$$H_A(D) = -\sum_{i=1}^n\frac{|D_i|}{|D|}\log_2\frac{|D_i|}{|D|}$$

### CART

基尼系数
: $$\textrm{Gini}(p) = \sum_{k=1}^K p_k(1-p_k) = 1 - \sum_{k=1}^K p_k^2$$

在特征 $A$ 下，集合 $D$ 的基尼系数为：
$$\textrm{Gini}(D, A) = \frac{|D_1|}{|D|}\textrm{Gini}(D_1) + \frac{|D_2|}{|D|}\textrm{Gini}(D_2)$$
其中，
$$D_1 = \{(x, y) \in D | A(x) = a\}$$
$$D_2 = D - D_1$$

## 回归树

### 最小二乘回归树

- 选择最优切分变量 $j$ 与最优切分点 $s$
  $$\min_{j,s}\left[\min_{c_1}\sum_{x_i \in R_1(j, s)}(y_i - c_1)^2 + \min_{c_2}\sum_{x_i \in R_2(j, s)}(y_i - c_2)^2\right]$$
  其中：
  $$R_1(j, s) = \{x | x^{(j)} \le s\}$$
  $$R_2(j, s) = \{x | x^{(j)} > s\}$$

## 剪枝

损失函数为：
$$C_{\alpha}(T) = C(T) + \alpha|T|$$
其中，
$$C(T) = \sum_{t=1}^{|T|}N_t H_t(T)$$
$|T|$ 为叶子节点的数量。
