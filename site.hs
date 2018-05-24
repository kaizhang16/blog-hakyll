{-# LANGUAGE OverloadedStrings #-}

import           Control.Monad.IO.Class   (liftIO)
import           Data.Monoid              ((<>))
import           Hakyll
import qualified Text.CSL                 as CSL
import           Text.CSL.Pandoc          (processCites)
import qualified Text.Pandoc              as P
import           Text.Pandoc.Builder      (str)
import qualified Text.Pandoc.CrossRef     as CR
import           Text.Pandoc.Highlighting (pygments)
import           Text.Pandoc.Shared       (eastAsianLineBreakFilter)


main :: IO ()
main =
  hakyll $ do
    match "css/*" $ do
      route idRoute
      compile compressCssCompiler
    match "images/*" $ do
      route idRoute
      compile copyFileCompiler
    match "js/*" $ do
      route idRoute
      compile copyFileCompiler
    match "posts/*" $ do
      route $ setExtension "html"
      compile $
        myPandocCompiler >>= saveSnapshot "content" >>=
        loadAndApplyTemplate "templates/post.html" postCtx >>=
        loadAndApplyTemplate "templates/default.html" postCtx >>=
        relativizeUrls
    match "index.html" $ do
      route idRoute
      compile $ do
        posts <- recentFirst =<< loadAll "posts/*"
        let indexCtx =
              listField
                "posts"
                (teaserField "teaser" "content" <> postCtx)
                (return posts) <>
              constField "title" "主页" <>
              defaultContext
        getResourceBody >>= applyAsTemplate indexCtx >>=
          loadAndApplyTemplate "templates/default.html" indexCtx >>=
          relativizeUrls
    match "templates/*" $ compile templateBodyCompiler


postCtx :: Context String
postCtx =
    dateField "date" "%Y-%m-%d" <>
    defaultContext

myPandocCompiler :: Compiler (Item String)
myPandocCompiler =
  pandocCompilerWithTransformM readerOptions writerOptions transformM

readerOptions :: P.ReaderOptions
readerOptions = defaultHakyllReaderOptions {P.readerExtensions = newExtensions}
  where
    defaultExtensions = P.readerExtensions defaultHakyllReaderOptions
    newExtensions = P.enableExtension P.Ext_emoji defaultExtensions

writerOptions :: P.WriterOptions
writerOptions = defaultHakyllWriterOptions

transformM :: P.Pandoc -> Compiler P.Pandoc
transformM p = unsafeCompiler $ do
  p' <- crossRef (eastAsianLineBreakFilter p)
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
