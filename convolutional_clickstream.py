
import tensorflow as tf
import pandas as pd #work with data as tables
import numpy as np #matrix multiplication library
import matplotlib.pyplot as plt #graph generator

#Perfiles:

	# Que la persona esta decidida a comprar:
	 # clicks rápidos, presiona botón comprar.

	# Que la persona esta averiguando de un producto.
	 # clicks rápidos, después lentos, muchos clicks.

	# Que la persona esta navegando y no tenga ese interés. 
	  # clicks lentos. Entrada y salida rápida.

# Vamos a decidir, según la velocidad de los 5 primeros clicks a qué perfil pertenece el usuario.

if__name__ = '__main__':
	run()

# Builing the model:
	
	#Step 0: download GA data using scitylana.com/

	#Step 1: load data using pandas library.

dataframe = pd.read_csv('data.csv') #dataframe object
dataframe = dataframe.drop(['','',''],axis=1) #drop columns we don't need
dataframe = dataframe[0:1000] #only first 1000 rows 

    #Step 2: add labels, define model

dataframe.loc[:,('y1')] = [1,1,1,0,1,0,1,1,1] #define good buy = 1, bad buy = 0.
dataframe.loc[:,('y2')] = dataframe['y1'] == 0 #y2 is a negation of y1
dataframe.loc[:,('y2')] = dataframe['y2'].astype(int)

    #Step 3: prepare data for TensoFlow (create matrixes)

inputX = dataframe.loc[:,['','']].as_metrix() #convert features into Tensors
inputY = dataframe.loc[:,['y1','y2']]

	#Step 4: write our our hyperparameters

learning_rate = 0.000001
training_epochs = 2000
display_step = 50
n_sample =  inputY.size

	#Step 5: create our computation graph/neural network

tf.placeholder(tf.float32, [None,32])

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

    #Step 3: Run triaing.

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

    #Step 4: Output results

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# Learns best fit is W: [0.1], b: [0.3]