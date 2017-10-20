module Main where

import CodeGeneration.Generation
import CodeGeneration.EvalFile
import Scheme.Types

import Control.Monad
import System.Environment

main :: IO ()
main = do
  path <- tail <$> getArgs
  let ps = displayProgram <$> generatePrograms
  results <- join <$> mapM getResults ps
  putStrLn $ writeResults results
