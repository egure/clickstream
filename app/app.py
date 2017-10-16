# -*- coding: utf-8 -*- 
#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.

#future for python 2 and 3
from __future__ import print_function
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template, request, jsonify
#scientific computing library for saving, reading, and resizing images

#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re
#mongo
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
#initalize our flask app
app = Flask(__name__)
import pandas as pd
from googleapiclient.errors import HttpError
from googleapiclient import sample_tools
from oauth2client.client import AccessTokenRefreshError
import argparse
import sys
import collections
import csv
import numpy as np
import pandas as pd 
import json 
        
#The DB connection will assume that the database has the same name as the Flask Appliction which is "app"
app = Flask(__name__)
# URI scheme for Cloud Storage.
GOOGLE_STORAGE = 'gs'
# URI scheme for accessing local files.
LOCAL_FILE = 'file'
brand = "Messiac"
url = "http://messiac.com"
company = ""
customer = company
	
#######################
#        VIEWS        #
#######################
@app.route('/')
def index():
	#render out pre-built HTML file right on the index page
	return render_template("index.html", brand = brand)

@app.route('/<company>/1112/prediktor/summary')
def summary(company):
    brand = "Messiac"
    path = company
    company = "CNN"
    url = "http://messiac.com"
    #os.system("python ../session_recorder/real_time.py;")
    return render_template("summary.html", brand = brand, company = company, url=url, customer=customer, path = path)

#######################
#    POST ANALYSIS    #
#######################
@app.route('/<company>/1112/prediktor/recommendations')
def recommendations(company):
    path = company
    return render_template("recommendations.html", brand = brand, path = path)

#######################
#  VIEW IN REAL TIME  #
#######################
@app.route('/<company>/1112/prediktor/real_time')
def real_time(company):
    path = company
        # Google Analytics API autentification
    from oauth2client.service_account import ServiceAccountCredentials
    # The scope for the OAuth2 request.
    SCOPE = 'https://www.googleapis.com/auth/analytics.readonly'
    # The location of the key file with the key data.
    KEY_FILEPATH = 'templates/neem.json'
    # Defines a method to get an access token from the ServiceAccount object.
    if company == "cnn":
        url = "http://cnn.cl/categoria/internacional"
    else:
        url = "https://www.avenida.com.ar/tienda-tecnologia"
    def get_access_token():
        ServiceAccountCredentials.from_json_keyfile_name(KEY_FILEPATH, SCOPE).get_access_token().access_token
    tojen = ServiceAccountCredentials.from_json_keyfile_name(KEY_FILEPATH, SCOPE).get_access_token().access_token
    return render_template("real_time.html", brand = brand, tojen = tojen, path = path, url = url)

#######################
#    HISTORIC VIEW    #
#######################
@app.route('/<company>/1112/prediktor/list')
def list(company):
    path = company
    return render_template("list.html", brand = brand, path = path)
	
#######################
#    TRAINING DATA    #
#######################
@app.route('/<company>/1112/prediktor/training_data')
def training_data(company):
    path = company
    return render_template("training_data.html", brand = brand, path = path)

#######################
# ENGAGEMENT-ANALYTICS#
#######################
@app.route('/<company>/1112/prediktor/analytics')
def analytics(company):
    path = company
        # Google Analytics API autentification
    from oauth2client.service_account import ServiceAccountCredentials
    # The scope for the OAuth2 request.
    SCOPE = 'https://www.googleapis.com/auth/analytics.readonly'
    # The location of the key file with the key data.
    KEY_FILEPATH = 'templates/neem.json'
    # Defines a method to get an access token from the ServiceAccount object.
    if company == "cnn":
        url = "http://cnn.cl/categoria/internacional"
    else:
        url = "https://www.avenida.com.ar/tienda-tecnologia"
    def get_access_token():
        ServiceAccountCredentials.from_json_keyfile_name(KEY_FILEPATH, SCOPE).get_access_token().access_token
    tojen = ServiceAccountCredentials.from_json_keyfile_name(KEY_FILEPATH, SCOPE).get_access_token().access_token
    return render_template("analytics.html", brand = brand, tojen = tojen, path = path, url = url)

#######################
#      PARAMETERS     #
#######################
@app.route('/<company>/1112/prediktor/parameters')
def parameters(company):
    path = company
    company = "CNN"
    return render_template("parameters.html", company = company, brand = brand, path = path)

#######################
#      EDIT USER      #
#######################
@app.route('/<company>/1112/prediktor/user')
def user(company):
    path = company
    company = "CNN"
    user_name = "Eduardo Castillo"
    return render_template("user.html", user_name = user_name, company = company, brand = brand, path = path)

#######################
#    CROSS SELLING    #
#######################
#Store the CVS file with the data matrix
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#######################
#    AJAX EXAMPLE     #
#######################
@app.route('/test', methods=['GET','POST'])
def test():
    return render_template("test.html", brand = brand)

@app.route('/test1', methods=['GET','POST'])
def test1():
    name =  request.form['name']
    email = request.form['email']
    print(request.form.get('registration'),request.form.get('name'),request.form.get('email'))
    if request.form.get('registration') == "success":
       return json.dumps({"abc":"successfuly registered"})

#######################
#   NUMPY  CLASSIFY   #
#######################
#input website data in the Neuralnetwork
from numpy import exp, array, random, dot
class NeuralNetwork():
    def __init__(self):
        #seed the random number generator, so it generates the same numbers 
        #every time the program runs
        random.seed(1)
        #we model a single neuron, with 3 input connection 
        #and 1 output connection we assign random weight
        #and mean 0
        self.synaptic_weights = 2 * random.random((3,1)) - 1
    #Activation function, which describes an s shape curve
    #we pass the weighted sum of the inputs through this function
    #to nroma
    def __sigmoid(self, x):  
        return 1/(1 + exp(-x))
    def __sigmoid_derivative(self,x):
        return x * (1-x)
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            #pass the training set through our weight times input
            output = self.predict(training_set_inputs)

            #calculate the error
            error = training_set_outputs - output

            #muliply the error by the input
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            #adjust the weights (backpropagate)
            self.synaptic_weights += adjustment
    def predict(self, inputs): 
        #input times weight, predict.
        return self.__sigmoid(dot(inputs,self.synaptic_weights))
#single layer feed forwards neural network
neural_network = NeuralNetwork()
print('Random starting synaptic weights:')
print(neural_network.synaptic_weights)
#training set: 4 exampleas, each consisting of 3 input values
# and 1 output values
training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_set_outputs = array([[0,1,1,0]]).T
#train the neural network using a training set
#do it 10,000 times and make small adjustments each time
neural_network.train(training_set_inputs, training_set_outputs, 10000)
print('New synaptic weights after training')
print(neural_network.synaptic_weights)
print('predicting')
print(neural_network.predict(array([1,0,0])))

#######################
#DEEP LEARNING PREDIC.#
#######################
@app.route('/predict/',methods=['GET','POST'])
def predict():
    from keras.models import Sequential #this is the NN model type 
    from keras.layers import Dense #layer types
    numpy.random.seed(7) #this is used for
    dataset = numpy.loadtxt("data/train_data.csv", delimiter=",")
     #dataset1 = numpy.loadtxt("data/63.csv")
     #Step 2: add labels, define model
     #var arr = Object.values(obj);
    X_train = dataset[:,0:8]
    print(X_train)
    X_test = dataset[:,0:8]
     # 1.0
    Y_train = dataset[:,8]
    Y_test = dataset[:,8]
     # 1.1
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
     # 2.1 compile the network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
     # 2.2 fit the network
    history = model.fit(X_train, Y_train, epochs=100, batch_size=10)
     # 2.3 evaluate the network
    loss, accuracy = model.evaluate(X_train, Y_train)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
     # 3. make predictions
     #probabilities = model.predict(X)
     #predictions = [float(round(x)) for x in probabilities]
     #accuracy = numpy.mean(predictions == Y)
     #print("Prediction Accuracy: %.2f%%" % (accuracy*100))
     #calculate predictions
    predictions = model.predict(X_test)
     #round predictions
    rounded = [round(x[0]) for x in predictions]
    probability_of_purchas_for_this_session = [x[0] for x in predictions]
    print(probability_of_purchas_for_this_session)    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True, threaded=True)
