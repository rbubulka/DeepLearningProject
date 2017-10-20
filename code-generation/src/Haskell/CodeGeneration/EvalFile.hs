
module CodeGeneration.EvalFile where

import Scheme.Types
import Scheme.ParseTypes
import Interpreter.Evaluate
import Interpreter.Types
import Scheme.Parse

import Data.List

newtype Valid =
  IsValid Bool
  deriving ( Show )

data Result
  = PE ParseError
  | EE EvalError
  | Val Value
  deriving ( Show )

data ResultLine
  = ResultLine String Result

instance Show ResultLine where
  show (ResultLine p (PE err)) =
    intercalate "," [ p
                    , "Nothing"
                    , show err
                    ]
  show (ResultLine p (EE err)) =
    intercalate "," [ p
                    , "Nothing"
                    , show err
                    ]
  show (ResultLine p (Val v)) =
    intercalate "," [ p
                    , displayValue v
                    , "Nothing"
                    ]

getResults :: String -> IO [ResultLine]
getResults content =
  mapM (\line ->
          ResultLine line <$> execLine line)
       (lines content)

writeResults :: [ResultLine] -> String
writeResults = intercalate "\n" . fmap show

execLine :: String -> IO Result
execLine line =
  case runParse line of
    Left err ->
      return $ PE err
    Right prog -> do
      val <- execEval prog
      case val of
        Left err ->
          return $ EE err
        Right v ->
          return $ Val v
