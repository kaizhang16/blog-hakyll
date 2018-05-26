---
title: 在 Hakyll 里使用 pandoc-citeproc 和 pandoc-crossref 
author: 张凯
tags: hakyll, pandoc, pandoc-citeproc, pandoc-crossref
---

本文介绍在 `Hakyll` 里使用 `pandoc-citeproc` 和 `pandoc-crossref` 的方法，这 2
个插件是 `Hakyll` 的重要优势。

<!--more-->

## `pandoc-citeproc`

`pandoc-citeproc` 支持参考文献的引用，如[@fig:pandocCiteprocDemo] 所示。

![`pandoc-citeproc` 示意](../images/pandoc-citeproc-demo.png){#fig:pandocCiteprocDemo}

## `pandoc-crossref`

`pandoc-crossref` 支持图、表和公式等的交叉引用，如[@fig:pandocCrossrefDemo] 所示。

![`pandoc-crossref` 示意](../images/pandoc-crossref-demo.png){#fig:pandocCrossrefDemo}

## 在 `Hakyll` 中使用 `pandoc-citeproc` 和 `pandoc-crossref`

```haskell
import           Hakyll
import qualified Text.CSL                 as CSL
import           Text.CSL.Pandoc          (processCites)
import qualified Text.Pandoc              as P
import           Text.Pandoc.Builder      (str)
import qualified Text.Pandoc.CrossRef     as CR

myPandocCompiler :: Compiler (Item String)
myPandocCompiler =
  pandocCompilerWithTransformM defaultHakyllReaderOptions defaultHakyllWriterOptions transformM

transformM :: P.Pandoc -> Compiler P.Pandoc
transformM p = unsafeCompiler $ do
  p' <- crossRef p
  processCites' p'

crossRef :: P.Pandoc -> IO P.Pandoc
crossRef = CR.runCrossRefIO meta Nothing CR.defaultCrossRefAction
  where
    meta =
      CR.figureTitle (str "图") <> CR.figPrefix (str "图.") <>
      CR.tableTitle (str "表") <>
      CR.tblPrefix (str "表.")

processCites' :: P.Pandoc -> IO P.Pandoc
processCites' p = do
  refs <- CSL.readBiblioFile "references/all.bib"
  style <- CSL.readCSLFile Nothing "csl/chinese-gb7714-2005-numeric.csl"
  return $ processCites style refs p
```
> 完整的 `site.hs` 在 <https://github.com/kaizhang91/blog/blob/master/site.hs>。
