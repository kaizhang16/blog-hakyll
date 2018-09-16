---
title: GPG 教程
author: 张凯
tags: Linux, GNU
---

本文介绍 GPG (GNU Privacy Guard) 的用法。

<!--more-->

## 基本操作

```
gpg --list-keys  # 列出公钥
gpg --gen-key  # 生成新密钥对

gpg --list-secret-keys --keyid-format LONG  # 列出私钥
gpg --armor --output ${filename} --export-secret-keys ${私钥 ID}  # 导出私钥
gpg --armor --export ${私钥 ID}  # 导出公钥
```
