---
title: 使用 Travis CI 部署 Github Pages
author: 张凯
tags: [travis-ci, gh-pages]
---

## Github Pages

`Github Pages` 是 `github` 免费提供的静态网站托管服务，我们可以在上面部署博客、
简历和项目介绍等。

## Travis CI

`Travis CI` 为 `github` 上的开源项目免费提供自动构建服务。比如，当从本地 `git
push` 代码到 `github` 时，会触发在 `Travis CI` 里定义的构建脚本。特别地，`Travis
CI` 支持部署到 `Github Pages`。例如我们可以编写下面的配置：

```yaml
deploy:
  provider: pages  # 表示 Github Pages
  local-dir: _site  # 要上传的文件夹
  skip-cleanup: true  # 必填，以免 Travis CI 删除要上传的文件
  github-token: $GITHUB_TOKEN  # 在 Github 的设置里获取
  keep-history: true
  on:
    branch: master  # 有更改 push 到 master 分支时触发构建
```

将之保存为 `.travis.yml`，并在
