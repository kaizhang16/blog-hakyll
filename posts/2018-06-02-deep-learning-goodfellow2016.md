---
title: 文献阅读：Deep Learning
author: 张凯
tags: 文献阅读, deep-learning
---

本文记录了《Deep Learning》[@goodfellow2016deep]的读后感。

<!--more-->

## Linear Algebra

### 范数

范数
: 满足以下性质的任意函数 $f$

  - $f(\bm{x}) = 0 \Rightarrow \bm{x} = \bm{0}$
  - $f(\bm{x} + \bm{y}) \le f(\bm{x}) + f(\bm{y})$ （三角不等式）
  - $\forall \alpha \in \mathbb{R}, f(\alpha\bm{x}) = |\alpha|f(\bm{x})$

$L^p$ 范数：
$$||\bm{x}||_p = \left(\sum_i |x_i|^p\right)^{\frac{1}{p}}$$ {#eq:Lp}
其中 $p \in \mathbb{R}, p \ge 1$。

$L^2$ 范数为 Euclidean 距离。

- $L^2$ 范数平方的优点
    - 计算简单
        - 偏微分只取决于 $x_i$，与其他 $x_j$ 没有关系
- $L^2$ 范数平方的缺点
    - 在原点附近变化缓慢

- $L^1$ 范数的优点
    - 在原点附近变化较快，可以区分 $0$ 和接近于 $0$ 的值

“$L^0$” 范数表示向量里非零元素的个数。（叫法不正确）

$L^\infty$ 范数或者最大范数：
$$||\bm{x}||_\infty = \max_i|x_i|$$ {#eq:LInfty}

Frobenius 范数为：
$$||\bm{A}|| = \sqrt{\sum_{i, j}A_{i, j}^2}$$ {#eq:frobeniusNorm}

点积可以用范数表示：
$$\bm{x}^T\bm{y} = ||\bm{x}||_2 ||\bm{y}||_2 \cos\theta$$ {#eq:dotProduct}

## Probability and Information Theory

### Bayes' Rule

$$P(\textrm{x}|\textrm{y}) = \frac{P(\textrm{x})P(\textrm{y}|\textrm{x})}{P(\textrm{y})}$$ {#eq:bayesRule}

### Gaussian Distribution

$$\mathcal{N}(x;\mu, \sigma^2) = \sqrt{\frac{1}{2\pi\sigma^2}}\exp\left(-\frac{1}{2\sigma^2}(x-\mu)^2\right)$$ {#eq:gaussianDistribution}
为了计算的简便，也可以用 $\beta^{-1} = \sigma^2$ 替代 $\sigma^2$ 作为参数，其中
$\beta \in (0, \infty)$，表示精度。

Gaussian 分布应用广泛的原因：

- 中心极限定理表明许多独立随机变量的和接近于正态分布；
- 同方差的所有分布里，正态分布的不确定性最大，意味着向模型插入了最少的先验知识。

$$\mathcal{N}(\bm{x};\bm{\mu},\bm{\Sigma}) = \sqrt{\frac{1}{(2\pi)^n\det(\bm{\Sigma})}}\exp\left(-\frac{1}{2}(\bm{x}-\bm{\mu})^\mathsf{T}\bm{\Sigma}^{-1}(\bm{x}-\bm{\mu})\right)$$ {#eq:gaussianDistributionMulti}

### 混合分布

$$P(\bm{\textrm{x}}) = \sum_{i}P(\textrm{c}=i)P(\bm{\textrm{x}} \mid \textrm{c}=i)$$ {#eq:distributionMixture}

Gaussian 混合
: $P(\bm{\textrm{x}} \mid \textrm{c}=i)$ 为 Gaussian 分布。

万能近似器
: Gaussian 混合。

### 常见函数

logistic sigmoid:
$$\sigma(x) = \frac{1}{1 + \exp(-x)}$$ {#eq:sigmoid}
因为取值范围是 $(0, 1)$，所以用来产生 Bernoulli 分布的 $\phi$ 参数。

softplus:
$$\zeta(x) = \log(1 + \exp(x))$$ {#eq:softplus}
因为取值范围是 $(0, \infty)$，所以用来产生正态分布的 $\beta$ 或 $\sigma$ 参数。
之所以叫 softplus 是因为它是
$$x^+ = \max(0, x)$$ {#eq:plus}
的光滑版。

Rectified Linear Unit(ReLU):
$$g(z) = \max(0, z)$$ {#eq:relu}
只要激活，梯度即为 $1$，相比于 sigmoid 更加容易学习。

tanh:
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$ {#eq:tanh}
$\tanh(0) = 0$，更类似于单位函数。

Radial basis function (RBF):
$$h_i = \exp\left(-\frac{1}{\sigma_i^2}\left\lVert\bm{W}_{:,i} - \bm{x}\right\rVert^2\right)$$ {#eq:rbf}
当 $\bm{x}$ 接近模板 $\bm{W}_{:,i}$ 时，函数激活；此外大部分情况不激活，所以难于
优化。

### 信息论

Kullback-Leibler (KL) 散度：
$$D_{\textrm{KL}}(P||Q) = \mathbb{E}_{\textrm{x} \sim P}\left[\log\frac{P(x)}{Q(x)}\right] = \mathbb{E}_{\textrm{x} \sim P}[\log P(x)-\log Q(x)]$$ {#eq:klDivergence}
描述了概率分布 $P(\textrm{x})$ 和 $Q(\textrm{x})$ 的区别有多大。KL 散度非负，当
且仅当 $P(\textrm{x})$ 和 $Q(\textrm{x})$ 处处相等[^klEqualEverywhere]时
$D_{\textrm{KL}}(P||Q) = 0$。但 KL 散度不对称：对于某些 $P$ 和 $Q$，
$D_{\textrm{KL}}(P||Q) \neq D_{\textrm{KL}}(Q||P)$。

交叉信息熵：
$$H(P, Q) = -\mathbb{E}_{\textrm{x} \sim P} \log Q(x)$$ {#eq:crossEntropy}
KL 散度和交叉信息熵之间的关系：$H(P, Q) = H(P) + D_{\textrm{KL}}(P||Q)$。

[^klEqualEverywhere]: 当 $\textrm{x}$ 为离散随机变量时，需要处处相等；当
    $\textrm{x}$ 为连续随机变量时，需要几乎处处相等。

## 数值计算

### 上溢和下溢

下溢
: 接近于 $0$ 的数取整为 $0$。

下溢
: 大数近似成 $\infty$ 或者 $-\infty$。

为了使数值计算稳定，需要把 softmax 里的 $x$ 换成 $x - \max_i x_i$。

### 病态条件

矩阵条件数

: $\kappa(\bm{A}) = \lVert \bm{A}^{-1}\rVert \cdot \lVert\bm{A}\rVert$。当范数选
  $L^2$ 范数，且 $\bm{A}$ 为正规矩阵时，$\kappa(\bm{A}) =
  \left\lvert\frac{\lambda_{max}(\bm{A})}{\lambda_{min}(\bm{A})}\right\rvert$。
  其中，$\lambda_{max}(\bm{A})$ 和 $\lambda_{min}(\bm{A})$ 分别为 $\bm{A}$ 的极
  大和极小（根据模数）特征值。当条件数较大时，矩阵求逆对输入错误非常敏感，此时称
  矩阵为病态条件的矩阵。

### 梯度之外：Jacobian 和 Hessian 矩阵

Jacobian 矩阵
: 对一个函数 $\bm{f}:\mathbb{R}^m \to \mathbb{R}^n$ 来说，
  $$\bm{J}_{i, j} = \frac{\partial}{\partial x_j} f(\bm{x})_i$$ {#eq:jacobianMatrix}
  为其 Jacobian 矩阵，$\bm{J} \in \mathbb{R}^{n\times m}$。

Hessian 矩阵
: 对一个函数 $f: \mathbb{R}^n \to \mathbb{R}$ 来说，
  $$\bm{H}(f)(\bm{x})_{i, j} = \frac{\partial^2}{\partial x_i\partial x_j}f(\bm{x})$$ {#eq:hessianMatrix}
  为其 Hessian 矩阵。

Hessian 矩阵的条件数描述了二阶导数有多大区别。

![Hessian 矩阵的条件数示意图](../images/hessian-matrix-condition-number.png){#fig:hessianMatrixConditionNumber}

### 有约束的优化

扩展 Lagrange 函数
: $$L(\bm{x}, \bm{\lambda}, \bm{\alpha}) = f(\bm{x}) + \sum_i \lambda_i g^{(i)}(\bm{x}) + \sum_j \alpha_j h^{(j)}(\bm{x})$$ {#eq:generalizedLagrange}

有约束的最小化
: $$\min_{\bm{x}}\max_{\bm{\lambda}}\max_{\bm{\alpha}, \bm{\alpha} \le 0}L(\bm{x}, \bm{\lambda}, \bm{\alpha})$$ {#eq:constrainedMin}

KKT 条件（有约束优化的最优点的必要条件）
: - 扩展 Lagrange 函数的梯度为 $0$；
 - 满足 $\bm{x}$ 和 KKT 乘子（$\bm{\lambda}, \bm{\alpha}$）的约束；
 - $\bm{\alpha} \odot \bm{h}(\bm{x}) = \bm{0}$。

活跃约束
: $h^{(i)}(\bm{x}^*) = 0$ 时称 $h^{(i)}(\bm{x})$ 为活跃的约束。

## Machine Learning Basics

### 学习算法

#### 例子：线性回归

均方误差
: $$\textrm{MSE}_{\textrm{test}} = \frac{1}{m}\sum_i (\hat{\bm{y}}^{(\textrm{test})} - \bm{y}^{(\textrm{test})})_i^2$$ {#eq:mse}

### 容量、过拟合和欠拟合

模型容量
: （非正式）拟合各类函数的能力。

表示容量
: 学习算法可以选择的函数家族。

有效容量
: 实际容量，小于表示容量。

最优容量
: 泛化误差最小时的模型容量。

### Maximum Likelihood Estimation（最大似然估计）

$$
\begin{align*}
\bm{\theta}_{\textrm{ML}} &= \arg\max_{\bm{\theta}}p_{\textrm{model}}(\mathbb{X;\bm{\theta}}) \\
&= \arg\max_{\bm{\theta}}\prod_{i=1}^m p_{\textrm{model}}(\bm{x}^{(i)};\bm{\theta}) \\
&= \arg\max_{\bm{\theta}}\sum_{i=1}^m \log p_{\textrm{model}}(\bm{x}^{(i)};\bm{\theta}) \\
&= \arg\max_{\bm{\theta}}\mathbb{E}_{\textrm{x} \sim \hat{p}_{\textrm{data}}} \log p_{\textrm{model}}(\bm{x};\bm{\theta})
\end{align*}
$$ {#eq:mle}
其中，$\mathbb{X} = \{\bm{x}^{(1)},\dots,\bm{x}^{(m)}\}$

最大似然估计等价于最小化交叉信息熵 $H(\hat{P}_{\textrm{data}},P_{\textrm{model}})$。

### Maximum A Posteriori (MAP) Estimation（最大后验概率估计）

$$
\begin{align*}
\bm{\theta}_{\textrm{MAP}} &= \arg\max_{\bm{\theta}}p(\bm{\theta}|\mathbb{X}) \\
&= \arg\max_{\bm{\theta}}\log p(\mathbb{X}|\bm{\theta})p(\bm{\theta}) \\
&= \arg\max_{\bm{\theta}}\left[\sum_{i=1}^m\log p(\bm{x}^{(i)}|\bm{\theta}) + \log p(\bm{\theta})\right]
\end{align*}
$$ {#eq:map}
先验分布为 $\mathcal{N}(\bm{w}; \bm{0}, \frac{1}{\lambda}\bm{I}^2)$ 时，先验项对
应于最大似然估计里的权重衰减。

### 非监督学习算法

低维表示
: 用小的表示压缩尽可能多的信息。

稀疏表示
: 大部分元素为 0。

独立表示
: 解耦数据分布的协方差以使各个维度统计独立。

### 构建一个机器学习算法

决策树和 k 平均不适用梯度优化，因为它们的损失函数有平坦区域。

## 深度前馈网络

MLPs
: Multilayer Perceptrons。

### 架构设计

万能近似定理
: 有一层线性输出层和至少一层任意激活函数的隐藏层的前馈网络，如果有足够多的隐藏单
  元，就可以以任意小的非零误差近似从一个有限维空间到另一个有限维空间的 Borel 可
  测函数。

深度矫正网络（ReLU？）表示的函数在单隐藏层网络里可能需要指数次隐藏单元。具体地，
深度矫正网络可以表示的线性区域数量为：
$$O\left(\binom{n}{d}^{d(l-1)}n^d\right)$$ {#eq:deepNetworkCapacity}
其中，$d$ 为输入层单元数量，$l$ 为深度，$n$ 为隐藏层单元数量。

### 历史说明

均方误差容易饱和，学习速度慢。

分段线性函数在某些点不可微，但学习速度快。

## 训练深度模型的优化

### 长期依赖

假设 $\bm{W}$ 有特征值分解 $\bm{W} =
\bm{V}\textrm{diag}(\bm{\lambda})\bm{V}^{-1}$，那么有
$$\bm{W}^t = (\bm{V}\textrm{diag}(\bm{\lambda})\bm{V}^{-1})^t = \bm{V}\textrm{diag}(\bm{\lambda})^t\bm{V}^{-1}$$ {#eq:gradientMultiply}
当 $\lvert\lambda_i\rvert > 1$ 时，梯度会爆炸，导致学习不稳定；当
$\lvert\lambda_i\rvert < 1$ 时，梯度会消失，导致不好判断向哪个方向优化。

### 参数初始化策略

梯度裁剪可以减缓梯度爆炸。

## 参考文献
