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
$$\mathcal{N}(\bm{x};\bm{\mu},\bm{\Sigma}) = \sqrt{\frac{1}{(2\pi)^n\det(\bm{\Sigma})}}\exp\left(-\frac{1}{2}(\bm{x}-\bm{\mu})^\mathsf{T}\bm{\Sigma}^{-1}(\bm{x}-\bm{\mu})\right)$$ {#eq:gaussianDistributionMulti}

### 分布的混合

$$P(\bm{\textrm{x}}) = \sum_{i}P(\textrm{c}=i)P(\bm{\textrm{x}} \mid \textrm{c}=i)$$ {#eq:distributionMixture}

Gaussian 混合
: $P(\bm{\textrm{x}} \mid \textrm{c}=i)$ 为 Gaussian 分布

万能近似器
: Gaussian 混合

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
