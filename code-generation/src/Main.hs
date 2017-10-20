module Main where

import CodeGeneration.Generation
import CodeGeneration.EvalFile
import Scheme.Types

import Control.Monad

main :: IO ()
main = do
  let ps = displayProgram <$> generatePrograms
  results <- join <$> mapM getResults ps
  putStrLn $ writeResults results
