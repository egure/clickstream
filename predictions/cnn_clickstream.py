#This CCN built to predict purchase intention

# 1 - On one side, it measures impulse [speed of the click stream]
# 2 - On the other hand,

from keras.models import Sequential #this is the NN model type 
from keras.layers import Dense #layer types
import numpy #matrix multiplication library
numpy.random.seed(7) #this is used for

# Builing the model:
  
  #Step 0: download Google dataset data using scitylana.com/

  # In this case we are going to use .csv dataset input

  #Step 1: load data using numpy.

dataset = numpy.loadtxt("data/train_data.csv", delimiter=",")

  #Step 2: add labels, define model

X_train = dataset[:,0:8]
X_test = dataset[:,0:8]

Y_train = dataset[:,8]
Y_test = dataset[:,8]

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

# calculate predictions
predictions = model.predict(X_test)
# round predictions
rounded = [round(x[0]) for x in predictions]
final = [x[0] for x in predictions]
print(final)