---
title: 使用 Travis CI 部署 Github Pages
author: 张凯
tags: [travis-ci, gh-pages]
---

## Github Pages

`Github Pages` 是 `Github` 免费提供的静态网站托管服务，我们可以在上面部署博客、
简历和项目介绍等。

## Travis CI

`Travis CI` 为 `Github` 上的开源项目免费提供自动构建服务。比如，当从本地 `git
push` 代码到 `Github` 时，会触发在 `Travis CI` 里定义的构建脚本。特别地，`Travis
CI` 支持部署到 `Github Pages`。首先，我们需要在
<https://github.com/settings/tokens> 生成一个 `Personal access token`，记为
`${githubToken}`。接着，将 `${githubToken}` 加密后写入 `.travis.yml`：

```
travis encrypt GITHUB_TOKEN=${githubToken} --add env.matrix
```

> 如果没有安装 `travis`，请先通过 `gem install travis` 安装。

然后在 `.travis.yml` 里添加如下内容：

```yaml
deploy:
  provider: pages  # 表示 Github Pages
  local-dir: _site  # 要上传的文件夹
  skip-cleanup: true  # 必填，以免 Travis CI 删除要上传的文件
  github-token: $GITHUB_TOKEN  # Github 的访问令牌
  on:
    branch: master  # master 分支有变化时触发部署
```

当然，还需要在 `.travis.yml` 填写 `language`、`install` 和 `script` 等字段指定如
何生成静态文件，例如：

```
language: nix
install:
- nix-channel --add https://nixos.org/channels/nixos-18.03 nixpkgs
- nix-channel --update
script:
- nix-build
- ./result/bin/site clean
- ./result/bin/site build
```

> 完整版 `.travis.yml` 参见
> <https://github.com/kaizhang91/blog/blob/master/.travis.yml>。

最后，在 <https://travis-ci.org/> 上关联 `Github` 工程。这样，当我们从本地 `git
push` 代码到 `Github` 上的 `master` 分支后，就会触发 `Travis CI` 的构建任务；构
建成功后，`Travis CI` 自动把构建成果（本例中为 `_site` 文件夹）推送到 `Github`
的 `gh-pages` 分支，我们就可以在 <https://${username}.github.io/${projectName}>
看到生成的静态网站了！
