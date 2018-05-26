---
title: 在 Hakyll 中使用 east_asian_line_breaks
author: 张凯
tags: hakyll, pandoc, east_asian_line_breaks
---

本文介绍 `Hakyll` 里中文断行的处理，这也是 `Hakyll` 相比于其他静态网站生成软件的
优势之一。

<!--more-->

## 中文断行问题

为了编辑和处理的方便，我们常常对 `Markdown` 源文件进行断行。比如
[@fig:lineBreakDemo] 对每 80 个字符进行断行。而默认模式下，`Pandoc` 把
`Markdown` 文件转换成 `HTML` 时，会把多行合并成一行，其中，断行用空格替代，效果
如[@fig:spaceDemo] 所示。在英文里，用空格分割单词，这么做没有问题。但是在中文里，
语句里不应该有空格。

![`Markdown` 源文件的断行示意图](../images/line-break-demo.png){#fig:lineBreakDemo}

![生成的 `HTML` 文件里的空格示意图](../images/space-demo.png){#fig:spaceDemo}

## `Pandoc` 的智能断行

大部分 `Markdown` 渲染软件都没有对中文断行做特殊处理，会影响渲染结果的美观性。而
`Pandoc` 提供了 `east_asian_line_breaks` 扩展，可以对东亚字符智能断行：

- 忽略段落里东亚字符之间的断行；
- 保留东亚字符与英文之间或英文单词之间的断行

效果如[@fig:eastAsianLineBreaksDemo] 所示。

![开启 `east_asian_line_breaks` 扩展后的 `HTML` 输出](../images/east-asian-line-breaks-demo.png){#fig:eastAsianLineBreaksDemo}

## 怎样在 `Hakyll` 里开启 `east_asian_line_breaks`？

我们可以定义自己的 `myPandocCompiler` 来编译 `Markdown` 源文件：

```
import           Hakyll
import           Text.Pandoc.Shared       (eastAsianLineBreakFilter)

myPandocCompiler :: Compiler (Item String)
myPandocCompiler =
  pandocCompilerWithTransform defaultHakyllReaderOptions defaultHakyllWriterOptions eastAsianLineBreakFilter
```

> 完整的 `site.hs` 在 <https://github.com/kaizhang91/blog/blob/master/site.hs>。

需要注意的是，自从 `pandoc-2.0` 之后，直接在 `ReaderOptions` 里激活
`Ext_east_asian_line_breaks` 并不会生效，原因参见
<https://github.com/jgm/pandoc/issues/3703>。
