from keras.models import Sequential, Model
from keras.layers.embedding import Embedding
from keras.layers import GRU
import tarfile
import numpy as np

#question answearnig system. LSTM doesn't work for long sequences of data.

#extract data. We use 
path = get_file('babu-task-v1.2.tar.gz')

#split data
train_stories = get_stories(tar.extractfile(path.format('train')))
test_data = get_stories(tar.extractfile(path.format('test')))

#vectorize data
#semantic memory:
#this input uses a GRU cell (instead of an LSTM) or data recovering unit
inputs_train, queries_train, answers_train = vectorize_stories(train_stories,word_idx,story_maxlen,query_maxlen) 
#episodic memory :
inputs_test, queries_test, answers_test = vetorize_stories(test_stories,word_idx,story_maxlen,query_maxlen)

#the hidden state of the input model, represents the input process in the vector. It outputs hidden layers after sentense or word vector (outputs are called facts).
#given a vector and previous time step vector, computes a current time 
#the update gate is a single layer neural network, we sum up the matrix multiplicatoins and add a bias term
#the update gate, can ignore the currente line.
#then comes the question module.
