module Lib
  ( myPandocCompiler
  , postCtx
  , postCtxWithTags
  ) where

import           Data.Monoid          ((<>))
import           Hakyll
import qualified Text.CSL             as CSL
import           Text.CSL.Pandoc      (processCites)
import qualified Text.Pandoc          as P
import           Text.Pandoc.Builder  (str)
import qualified Text.Pandoc.CrossRef as CR
import           Text.Pandoc.Shared   (eastAsianLineBreakFilter)

postCtx :: Context String
postCtx =
    dateField "date" "%Y-%m-%d" <>
    defaultContext

postCtxWithTags :: Tags -> Context String
postCtxWithTags tags = tagsField "tags" tags <> postCtx

myPandocCompiler :: Compiler (Item String)
myPandocCompiler =
  pandocCompilerWithTransformM readerOptions writerOptions transformM

readerOptions :: P.ReaderOptions
readerOptions = defaultHakyllReaderOptions {P.readerExtensions = newExtensions}
  where
    defaultExtensions = P.readerExtensions defaultHakyllReaderOptions
    newExtensions = P.enableExtension P.Ext_emoji defaultExtensions

writerOptions :: P.WriterOptions
writerOptions =
  defaultHakyllWriterOptions
    { P.writerHTMLMathMethod =
        P.MathJax
          "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"
    , P.writerTableOfContents = True
    , P.writerTemplate = Just "$body$\n<div id=\"toc\">$toc$</div>"
    }

transformM :: P.Pandoc -> Compiler P.Pandoc
transformM p = do
  p' <- crossRef (eastAsianLineBreakFilter p)
  processCites' p'

crossRef :: P.Pandoc -> Compiler P.Pandoc
crossRef p =
  unsafeCompiler $ do
    let meta =
          CR.figureTitle (str "图") <> CR.figPrefix (str "图.") <>
          CR.tableTitle (str "表") <>
          CR.tblPrefix (str "表.")
    CR.runCrossRefIO meta Nothing CR.defaultCrossRefAction p

processCites' :: P.Pandoc -> Compiler P.Pandoc
processCites' p = do
  style <-
    unsafeCompiler $
    CSL.readCSLFile Nothing "csl/chicago-author-date.csl"
  bib <- load $ fromFilePath "references/all.bib"
  let Biblio refs = itemBody bib
  return $ processCites style refs p
