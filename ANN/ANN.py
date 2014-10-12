import numpy as np
from scipy.special import expit as sigmoid


class ANN:
    def __init__(self):
        self._num_input = 0
        self._num_hidden = 0
        self._num_output = 0
        self._learning_rate = 0
        self._hidden_layer_weights = []
        self._output_layer_weights = []
        self._input_layer = []  # input layer has to be 1 X ? array
        self._hidden_layer = []  #
        self._output_layer = []
        self._actual_output = []

    @property
    def num_input(self):
        return self._num_input

    @num_input.setter
    def num_input(self, num):
        self._num_input = num

    @property
    def num_hidden(self):
        return self._num_hidden

    @num_hidden.setter
    def num_hidden(self, num):
        self._num_hidden = num

    @property
    def num_output(self):
        return self._num_output

    @num_output.setter
    def num_output(self, num):
        self._num_output = num

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, num):
        self._learning_rate = num

    @property
    def hidden_layer_weights(self):
        return self._hidden_layer_weights

    @property
    def output_layer_weights(self):
        return self._output_layer_weights

    @property
    def input_layer(self):
        return self._input_layer

    @property
    def hidden_layer(self):
        return self._hidden_layer

    @property
    def output_layer(self):
        return self._output_layer

    @property
    def actual_output(self):
        return self._actual_output

    def setUp(self):
        #Create hidden layer weight matrix
        # self._hidden_layer_weights = np.random.uniform(-1, 1, (self.num_input + 1, self.num_hidden))
        self._hidden_layer_weights = [[0.1008513], [0.1008513], [0.1]]
        #Create output layer weight matrix
        # self._output_layer_weights = np.random.uniform(-1, 1, (self.num_hidden + 1, self.num_output))
        self._output_layer_weights = [[0.1343929], [0.1189104]]

    def feed_forward(self, inputvalue):
        self._input_layer = np.insert(inputvalue, 0, 1, axis=1)
        self._hidden_layer = np.insert(sigmoid(np.dot(self._input_layer, self._hidden_layer_weights)), 0, 1, axis=1)
        self._output_layer = sigmoid(np.dot(self._hidden_layer, self._output_layer_weights))

    def back_propagate(self, actual_output):
        output_layer_error = self.calculate_output_layer_error(actual_output)
        hidden_layer_error = self.calculate_hidden_layer_error(output_layer_error)
        self.update_output_layer_weights(output_layer_error)

        self.update_hidden_layer_weights(hidden_layer_error)

    def calculate_output_layer_error(self, actual_output):
        output_layer_error = np.multiply(np.multiply((1 - self._output_layer), self._output_layer),
                                         (actual_output - self._output_layer))
        print "output_layer_error"
        print output_layer_error
        return output_layer_error

    def calculate_hidden_layer_error(self, output_layer_error):
        error_sum = np.dot(output_layer_error, np.transpose(self._output_layer_weights))
        print "error_sum"
        print error_sum
        hidden_layer_error = np.multiply(np.multiply((1 - self._hidden_layer), self._hidden_layer), error_sum)
        print "hidden_layer_error"
        print hidden_layer_error
        return hidden_layer_error

    def update_output_layer_weights(self, output_layer_error):
        delta = np.dot(np.transpose(self._hidden_layer), output_layer_error) * self.learning_rate
        self._output_layer_weights = np.add(self._output_layer_weights, delta)
        print "_output_layer_weights"
        print self._output_layer_weights

    def update_hidden_layer_weights(self, hidden_layer_error):
        delta = np.dot(np.transpose(self._input_layer), np.delete(hidden_layer_error, 0, axis=1)) * self.learning_rate
        self._hidden_layer_weights = np.add(self._hidden_layer_weights, delta)
        print "_hidden_layer_weights"
        print self._hidden_layer_weights


ann = ANN()
ann.num_input = 2
ann.num_hidden = 1
ann.num_output = 1
ann.learning_rate = 0.3
ann.setUp()
inputvalue = [[0, 1]]
ann.feed_forward(inputvalue)
ann.back_propagate([[0]])
print ann.output_layer
print ann.hidden_layer
print ann.input_layer
# output_layer_error = ann.calculate_output_layer_error([[1,0]])
# hidden_layer_error = ann.calculate_hidden_layer_error(output_layer_error)
# ann.update_output_layer_weights(output_layer_error)
# ann.update_hidden_layer_weights(hidden_layer_error)
# print output_layer_error
# print hidden_layer_error
