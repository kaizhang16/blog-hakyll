---
title: "文献阅读：Understanding LSTM Networks"
author: 张凯
tags: 文献阅读, deep-learning
---

本文记录了 [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) 的读后感。

<!--more-->

## LSTM

- 遗忘门：$$f_{t} = \sigma(W_{f}\cdot [h_{t-1}, x_{t}] + b_f)$$
- 输入门：$$i_{t} = \sigma(W_{i}\cdot [h_{t-1}, x_{t}] + b_i)$$
- 新的候选 cell：$$\tilde{C}_{t} = \tanh(W_{C}\cdot[h_{t-1},x_{t}] + b_C)$$
- 新 cell：$$C_{t} = f_{t}\odot C_{t-1} + i_{t}\odot\tilde{C}_{t}$$
- 输出门：$$o_{t} = \sigma(W_{o}\cdot[h_{t-1},x_{t}] + b_o)$$
- 输出：$$h_t = o_t \odot \tanh(C_t)$$

## GRU

- 更新门：$$z_{t} = \sigma(W_{z}\cdot [h_{t-1}, x_{t}])$$
- 重置门：$$r_{t} = \sigma(W_{r}\cdot [h_{t-1}, x_{t}])$$
- 候选输出：$$\tilde{h}_{t} = \tanh(W\cdot [r_t\odot h_{t-1}, x_{t}])$$
- 输出：$$h_{t} = (1 - z_{t})\odot h_{t-1} + z_{t}\odot\tilde{h}_{t}$$

## 参考文献
