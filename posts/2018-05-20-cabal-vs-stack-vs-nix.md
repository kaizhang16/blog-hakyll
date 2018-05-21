---
title: Cabal vs Stack vs Nix
author: 张凯
---

指标           cabal     stack  nix
------------   --------  -----  -----
隔离性         ✘         ✔      ✔
免编译依赖     ✘         ✘      ✔
支持静态链接库 ✔         ✔      ✘
占用空间少     ✔         ✘      ✔
支持系统依赖   ✘         ✘      ✔
支持镜像源     ✔         ✔      ✘

: `cabal` vs `stack` vs `nix` {#tbl:cabalStackNix}
