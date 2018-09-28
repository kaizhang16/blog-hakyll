---
title: 初始化 Arch Linux
author: 张凯
tags: Linux, Arch Linux
---

本文介绍了初始化 Arch Linux 的过程

<!--more-->

## AUR

### yay

```
```

## 编辑

### Fonts

```
sudo pacman -S adobe-source-han-sans-cn-fonts adobe-source-han-serif-cn-fonts
sudo pacman -S adobe-source-code-pro-fonts wqy-zenhei 
sudo pacman -S ttf-dejavu ttf-inconsolata ttf-roboto
```

### Spacemacs

#### Haskell

```
sudo pacman -S hlint stylish-haskell hasktags hoogle
yay haskell-apply-refactor
stack install intero
```

### Pandoc

```
sudo pacman -S graphviz
sudo pacman -S pandoc-citeproc pandoc-crossref
sudo pacman -S texlive-most
sudo pacman -S texlive-lang
```

## Haskell

```
sudo pacman -S haskell-hakyll
```
