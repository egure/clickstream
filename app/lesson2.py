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

print 'Random starting synaptic weights:'
print neural_network.synaptic_weights

#training set: 4 exampleas, each consisting of 3 input values
# and 1 output values
training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_set_outputs = array([[0,1,1,0]]).T

#train the neural network using a training set
#do it 10,000 times and make small adjustments each time
neural_network.train(training_set_inputs, training_set_outputs, 10000)

print 'New synaptic weights after training'
print neural_network.synaptic_weights

print 'predicting'
print neural_network.predict(array([1,0,0]))