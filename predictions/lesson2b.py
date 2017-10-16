import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#load data
#dataframe object which is easy to parse
dataframe = pd.read_csv('data/data.csv')
dataframe = dataframe.drop(['index','price','sq_price'], axis=1)
dataframe = dataframe[0:10]

#step 2 - add labels
#1 is good buy and 0 is bad buy
dataframe.loc[:, ('y1')] = [1,1,1,0,0,1,0,1,1,1]
dataframe.loc[:, ('y2')] = dataframe['y1'] == 0
#y2 - 0 is good buy and 1 is bad buy
dataframe.loc[:, ('y2')] = dataframe['y2'].astype(int)
print(dataframe)

#step 3 shape it and feed it into tensorflow
inputX = dataframe.loc[:,['area','bathrooms']].as_matrix()
#convert labels to inputs 
inputY = dataframe.loc[:,['y1','y2']].as_matrix()

print(inputY)

#step 4 - Write out our parameters. If the prediction isn't accurate you can adjust this parameters.
learning_rate = 0.000001 #if the rate is higher it shows more
training_epochs = 2000 #repetitions
display_step = 50 #is to show the progression
n_samples = inputY.size

#step 5 - create computation graph
#for feature input tensors, none means any numbers of examples, is the batch size
x = tf.placeholder(tf.float32,[None,2])

b = tf.Variable(tf.zeros([2]), name="b")

#create weights 
W = tf.Variable(tf.zeros([2,2]))

#multipy our weights by our inputs, first calculation
#weights are how we govern how data flows in our computation graph

#y_value es el valor de la funcion (valor esperado segun la funcion)
y_values = tf.add(tf.matmul(x, W), b) 

#Step 6 perform training
y = tf.nn.softmax(y_values)
y_ = tf.placeholder(tf.float32, [None, 2])

#Step 6, perform training
#create our cost function, mean squared error
#reduce error - sum computes the elements across dimensinos of a tensor
cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)
#Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#initialize varibles as TF needs
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#training loop
for i in range(training_epochs):
	sess.run(optimizer, feed_dict={x: inputX, y_: inputY})
	#write out logs of training
	if(i) % display_step == 0:
		cc = sess.run(cost, feed_dict={x: inputX, y_: inputY})
		print "Traning step"