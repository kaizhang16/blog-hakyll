---
title: 深度学习的 Softmax
author: 张凯
tags: deep-leading, softmax
---

本文介绍深度学习中的 `Softmax`。

<!--more-->

`Softmax` 将输入向量映射成概率分布，具体为[@goodfellow2016deep]：

$$ \mathrm{softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_{j}\exp(z_j)} $$ {#eq:softmax}

其中，

$$ \mathbf{z} = \mathbf{W}^T\mathbf{h} + \mathbf{b}$$ {#eq:linear}

## 参考文献
