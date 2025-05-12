import fastapi
import gensim
import pyvi
import sklearn
import nltk
import numpy
import pydantic
import uvicorn

print("fastapi:", fastapi.__version__)
print("gensim:", gensim.__version__)
from importlib.metadata import version
print("pyvi:", version("pyvi"))
print("sklearn:", sklearn.__version__)
print("nltk:", nltk.__version__)
print("numpy: ", numpy.__version__)
print("pydantic:", pydantic.__version__)
print ("uvicorn: ", uvicorn.__version__)
