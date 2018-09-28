---
title: 初始化 Arch Linux
author: 张凯
tags: Linux, Arch Linux
---

本文介绍了初始化 Arch Linux 的过程

<!--more-->

## 安装

### 时区

```
ln -sh /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
```

### boot

```
pacman -S grub
grub-install /dev/sda
grub-mkconfig -o /boot/grub/grub.cfg
```

## 基础

### 工具

```
systemctl start dhcpcd
systemctl enable dhcpcd
pacman -S adobe-source-han-sans-cn-fonts adobe-source-han-serif-cn-fonts
pacman -S adobe-source-code-pro-fonts
pacman -S base-devel dmenu emacs
pacman -S fcitx fcitx-cloudpinyin fcitx-googlepinyin
pacman -S fd feh fish fzf
pacman -S git go gopass openssh parcellite python stack sudo
pacman -S ripgrep rust rxvt-unicode tmux trayer
pacman -S ttf-dejavu ttf-font-awesome ttf-inconsolata ttf-roboto
pacman -S variety vim wqy-microhei wqy-zenhei
pacman -S xclip xmobar xmonad xorg-server xorg-xinit
pacman -S xsel z
```

### 用户

```
useradd --create-home kai
passwd kai
gpasswd -a kai wheel
visudo  # 让 wheel 拥有 sudo 权限
chsh -s /usr/bin/fish kai
```

### Virtual Box

```
sudo pacman -S virtualbox-guest-utils  # 选择 virtualbox-guest-modules-arch
sudo gpasswd --add kai vboxsf
sudo systemctl start vboxservice
sudo systemctl enable vboxservice
```

### yay

```
git clone https://aur.archlinux.org/yay.git
cd yay
makepkg -si
```

### 常用 AUR 软件

```
yay google-chrome
```

## 配置

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

### Haskell

```
sudo pacman -S haskell-hakyll
```
