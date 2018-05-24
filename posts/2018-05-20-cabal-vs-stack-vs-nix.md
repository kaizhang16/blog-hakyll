---
title: Cabal vs Stack vs Nix
author: 张凯
tags: [haskell, cabal, stack, nix]
---

`Cabal`、`Stack` 与 `Nix` 的优缺点比较。

<!--more-->

指标           cabal              stack               nix
------------   -----------------  ------------------  -----
隔离性         ✘                  :heavy_check_mark:  :heavy_check_mark:
免编译依赖     ✘                  ✘                   :heavy_check_mark:
支持静态链接库 :heavy_check_mark: :heavy_check_mark:  ✘
占用空间少     :heavy_check_mark: ✘                   :heavy_check_mark:
支持系统依赖   ✘                  ✘                   :heavy_check_mark:
支持镜像源     :heavy_check_mark: :heavy_check_mark:  ✘

: `cabal` vs `stack` vs `nix` {#tbl:cabalStackNix}
