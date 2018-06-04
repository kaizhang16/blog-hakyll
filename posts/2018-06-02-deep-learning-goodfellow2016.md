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

## Machine Learning Basics

### Maximum Likelihood Estimation（最大似然估计）

$$
\begin{align*}
\bm{\theta}_{\textrm{ML}} &= \arg\max_{\bm{\theta}}p_{\textrm{model}}(\mathbb{X;\bm{\theta}}) \\
&= \arg\max_{\bm{\theta}}\prod_{i=1}^m p_{\textrm{model}}(\bm{x}^{(i)};\bm{\theta}) \\
&= \arg\max_{\bm{\theta}}\sum_{i=1}^m \log p_{\textrm{model}}(\bm{x}^{(i)};\bm{\theta})
\end{align*}
$$ {#eq:mle}
其中，$\mathbb{X} = \{\bm{x}^{(1)},\dots,\bm{x}^{(m)}\}$

### Maximum A Posteriori (MAP) Estimation（最大后验概率估计）

$$
\begin{align*}
\bm{\theta}_{\textrm{MAP}} &= \arg\max_{\bm{\theta}}p(\bm{\theta}|\mathbb{X}) \\
&= \arg\max_{\bm{\theta}}\log p(\mathbb{X}|\bm{\theta})p(\bm{\theta}) \\
&= \arg\max_{\bm{\theta}}\left[\sum_{i=1}^m\log p(\bm{x}^{(i)}|\bm{\theta}) + \log p(\bm{\theta})\right]
\end{align*}
$$ {#eq:map}

## 参考文献
