{-# LANGUAGE OverloadedStrings #-}

module Main where

import           Data.Monoid              ((<>))
import           Hakyll
import           Hakyll.Web.Sass          (sassCompiler)
import qualified Lib                      as L


main :: IO ()
main =
  hakyll $ do
    match "css/*.scss" $ do
      route $ setExtension "css"
      compile (fmap compressCss <$> sassCompiler)
    match "images/*" $ do
      route idRoute
      compile copyFileCompiler
    match "js/*" $ do
      route idRoute
      compile copyFileCompiler
    match "references/*" $ do
      route idRoute
      compile biblioCompiler
    tags <- buildTags "posts/*" (fromCapture "tags/*.html")
    tagsRules tags $ \tag pattern' -> do
      let title = tag
      route idRoute
      compile $ do
        posts <- recentFirst =<< loadAll pattern'
        let ctx =
              listField
                "posts"
                (teaserField "teaser" "content" <> L.postCtxWithTags tags)
                (return posts) <>
              constField "title" title <>
              defaultContext
        makeItem "" >>= loadAndApplyTemplate "templates/tag.html" ctx >>=
          loadAndApplyTemplate "templates/default.html" ctx >>=
          relativizeUrls
    match "posts/*" $ do
      route $ setExtension "html"
      compile $
        L.myPandocCompiler >>= saveSnapshot "content" >>=
        loadAndApplyTemplate "templates/post.html" (L.postCtxWithTags tags) >>=
        loadAndApplyTemplate "templates/default.html" (L.postCtxWithTags tags) >>=
        relativizeUrls
    match "index.html" $ do
      route idRoute
      compile $ do
        posts <- recentFirst =<< loadAll "posts/*"
        let indexCtx =
              listField
                "posts"
                (teaserField "teaser" "content" <> L.postCtxWithTags tags)
                (return posts) <>
              constField "title" "主页" <>
              defaultContext
        getResourceBody >>= applyAsTemplate indexCtx >>=
          loadAndApplyTemplate "templates/default.html" indexCtx >>=
          relativizeUrls
    match "templates/*" $ compile templateBodyCompiler
