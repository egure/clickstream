#goal is to create word vectors from game of thrones dataset
#and analyze them to see semantic similarity

from  __future__ import absolute_import, division, print_function

import codecs #for word encoding
import glob #regex
import multiprocessing #concurrency
import os #for open a file
import pprint #make human readable
import re #regular expresions
import nltk #natural language analyser
import gesin.models.word2vps as v2c #word vectors
import sklearn.mainfold #dimensionality reduction (buscar video)
import numpy as numpy #mathlibrary
import matplotlib.py as plt #plotting
import pandas as pd 
import seaborn as sns #visualization

#process dataset (clean our data)
nltk.download('punkt') #pretrained toekanizer
nltk.download('stopwords') #words like: the, an, a, of
book_filenames = sorted(glob.glob("./*.txt")) #bring books using glob
book_filenames


