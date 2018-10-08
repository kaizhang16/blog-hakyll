---
title: 配置 Arch Linux
author: 张凯
tags: Linux, Arch Linux
---

本文介绍了初始化 Arch Linux 的过程

<!--more-->

## 安装

### 分区

分区方案：

- `/`: 20 GB
- `/var`: 15 GB
- `/home`: 45 GB

```
fdisk /dev/sda  # 选 dos 分区表

mkfs.ext4 /dev/sda1
```

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
pacman -S base-devel deepin-screenshot dmenu emacs
pacman -S fcitx fcitx-cloudpinyin fcitx-googlepinyin
pacman -S fcitx-im  # 选择全部
pacman -S fd feh fish fzf
pacman -S git go gopass mplayer pavucontrol net-tools
pacman -S openssh parcellite python stack sudo
pacman -S ripgrep rust rxvt-unicode tmux trayer
pacman -S ttf-dejavu ttf-font-awesome ttf-inconsolata ttf-roboto
pacman -S variety vim wqy-microhei wqy-zenhei
pacman -S xclip xmobar xmonad xorg-server xorg-xinit
pacman -S xsel z zsh
```

### 用户

```
useradd --create-home kai
passwd kai
gpasswd -a kai wheel
visudo  # 让 wheel 拥有 sudo 权限
chsh -s /usr/bin/zsh kai
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

### zsh

```
git clone https://github.com/Tarrasch/antigen-hs.git ~/.zsh/antigen-hs/
# 修改 stack.yaml 里的 resolver 版本
zsh  # Yes, Yes ...
# 更新 MyAntigen.hs 后，请 antigen-hs-setup
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

```
sudo pacman -S uget
sudo pacman -S goldendict
```
