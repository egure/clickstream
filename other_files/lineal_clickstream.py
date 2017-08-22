
import tensorflow as tf
import pandas as pd #work with data as tables
import numpy as np #matrix multiplication library

# We will decide if the customer fit is likely to make a purchase, by measuring the speed of his clickstream.
# We will train our model, to predict the customer intention after 5 clicks. 
def main():
	if __name__ == '__main__':
		main()

# Builing the model:
	
	#Step 0: for this excercise we are using data.csv file in the ~/data folder
	#The final product will input session information on a .csv file imported from Google Analytics, using scitylana.com/

	#Step 1: load data from .csv file using pandas library.

dataframe = pd.read_csv('data/data.csv') #dataframe object
dataframe = dataframe.drop([],axis=1) #drop columns we don't need

    #Step 2: add labels, define model

dataframe.loc[:,('y2')] = dataframe['y1'] == 0  #y2 is a negation of y1
dataframe.loc[:,('y2')] = dataframe['y2'].astype(int)
dataframe
    #Step 3: prepare data for TensoFlow (create matrixes)

inputX = dataframe.loc[:, ['velocidad_1', 'velocidad_2','velocidad_3', 'velocidad_4', 'velocidad_5']].as_matrix() #convert features into Tensors
inputY = dataframe.loc[:, ['y1','y2']].as_matrix() #y1 is define on the data input

	#Step 4: write our our hyperparameters

learning_rate = 0.000001
training_epochs = 2000
display_step = 50
n_samples =  inputY.size

	#Step 5: create our computation graph/neural network
	# none means any number of examples, 2 because we have 2 features and it's a 2d matrix

x = tf.placeholder(tf.float32, [None,5]) #placeholder are gatewyas for data infro computational graph
W = tf.Variable(tf.zeros([5,2])) #tf variable holds parameters
b = tf.Variable(tf.zeros(2)) #add bias adjusts the linear reggression

	#multiply our weight by our inputs and add biases

y_values = tf.add(tf.matmul(x, W), b)

#apply softmax to the function. It nomalizes our values.
y = tf.nn.softmax(y_values)

#feed in a matrix of labels
y_ = tf.placeholder(tf.float32, [None,2])

	#Step 6 perform training
	#create our cost function, mean squared error

cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)

#Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Minimize the mean squared errors.

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()
sess = tf.Session() # Launch the graph.
sess.run(init)

    #Step 7: Output results

for i in range(training_epochs):
	sess.run(optimizer, feed_dict={x: inputX, y_:inputY})  #performing optimizer which is gradient descent

	#write out logs of training
	if(i) % display_step == 0:
		cc = sess.run(cost, feed_dict={x:inputX,y_:inputY})
		print "Training step:",'%04d' % (i), "Cost:", "{:.9f}".format(cc)

print "Optimization Finished!"
training_cost = sess.run(cost, feed_dict={x: inputX, y_:inputY})
print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b)


print "The probability of sale completion for each item in the list is:"
boom = sess.run(y, feed_dict = {x: inputX})
print boom