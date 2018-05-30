---
title: "文献阅读：Dropout as a Bayesian Approximation: Insights and Applications"
author: 张凯
tags: deep-learning, 文献阅读, dropout, bayesian
---

本文记录了《Dropout as a Bayesian Approximation: Insights and Applications》的读后感。

<!--more-->

## Dropout 为什么能缓解过拟合[@gal2015dropout]？

$\mathbf{W}$ 的先验分布：

$$ p(\mathbf{W}) \sim \mathcal{N}(0, \mathbf{I}_K) $$ {#eq:pW}

$\mathbf{W}$ 的后验分布：

$$ q(\mathbf{W}) = \prod\limits_{q=1}^Q q(\mathbf{w_q}) $$ {#eq:qW}
$$ q(\mathbf{w}_q) = p_1\mathcal{N}(\mathbf{m}_q, \sigma^2 \mathbf{I}_K) + (1 - p_1)\mathcal{N}(0, \sigma^2 \mathbf{I}_K) $$

## 参考文献
