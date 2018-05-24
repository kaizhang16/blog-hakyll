---
title: 使用 Nix 编译 Haskell 工程
author: 张凯
tags: [haskell, nix]
---

本文介绍如何使用 `Nix` 来编译 `Haskell` 工程。

<!--more-->

## 什么是 `Nix`？

`Nix` 是一个通用的包管理工具，对 `Haskell` 提供了特别好的支持[^supportHaskell]。

## 为什么要使用 `Nix`？

目前 `Stack` 是非常流行的 `Haskell` 包管理工具，很好地实现了编译的隔离性和可复现
性。那为什么还要学习和使用 `Nix` 呢？因为 `Stack` 需要在本地编译依赖，这导致了 2
个问题：

- 第一次编译速度较慢
- 需要较大内存[^bigMemory]
- `resolver` 版本更新频繁，共存多个版本的话占用较大硬盘空间

而 `Travis CI` 有如下限制：

- 编译时间不能超过 50 分钟
- 容器提供 4 GB 内存，虚拟机提供 8 GB 内存，对 `Stack` 来说资源较紧张

对于一些依赖较多的工程来说，在 `Travis CI` 里编译就变得不可行了。而 `Nix` 恰恰克
服了 `Stack` 的上述缺点：

- 不需要本地编译，直接从 <https://cache.nixos.org> 下载二进制缓存

而 `Travis CI` 的下载速度特别快，可以达到几十 MB 乃至百 MB。因此对于不需要静态链
接的 `Haskell` 工程，在 `Travis CI` 里用 `Nix` 编译是一个很好的选择。

## 如何使用 `Nix` 编译 `Haskell` 工程？

首先，安装 `Nix`：

```
bash <(curl https://nixos.org/nix/install)
```

接着，安装 `cabal2nix`：

```
nix-env -i cabal2nix
```

然后借助 `cabal2nix` 生成编译配置:

```
cd $project
cabal2nix . > foo.nix
```

再配置 `default.nix`：

```nix
let
  pkgs = import <nixpkgs> { };

in
  rec {
    foo = pkgs.haskellPackages.callPackage ./foo.nix { };
  }
```

现在可以编译了！

```
nix-build
```

[^bigMemory]: 较大的项目需要 8 GB。
[^supportHaskell]: 提供了所有在 `Hackage` 上注册的 `Haskell` 库
