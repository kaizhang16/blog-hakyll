---
title: Cabal vs Stack vs Nix
author: 张凯
---

# 测试

指标           cabal     stack  nix
------------   --------  -----  -----
隔离性         不好      好     好
是否编译依赖   是        是     是
支持静态链接库 是        是     否
占用空间       小        大     小
支持系统依赖   否        否     否

Table: `cabal` vs `stack` vs `nix`
