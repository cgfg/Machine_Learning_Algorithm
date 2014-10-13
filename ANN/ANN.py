import os

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
        #
        self.training_data = []
        self.testing_data = []
        self.validation_data = []
        self.attributes = []
        self.classes = []
        self.separator = ' '
        self.attributes_index_list = []

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

    @property
    def get_training_data(self):
        return self.training_data

    @property
    def get_testing_data(self):
        return self.testing_data

    @property
    def get_validation_data(self):
        return self.post_pruning_data

    @property
    def get_attributes(self):
        return self.attributes

    @property
    def get_classes(self):
        return self.classes

    def setUp(self, data_name, result_id):
        cur_dir = os.path.dirname(__file__)  # Get current script file location
        testing = self.load_testing_data(cur_dir + '/data/' + data_name + '_results_testing_data_' + str(result_id))
        training = self.load_training_data(cur_dir + '/data/' + data_name + '_results_training_data_' + str(result_id))
        validation = self.load_validation_data(
            cur_dir + '/data/' + data_name + '_results_pruning_data_' + str(result_id))
        # Load attributes
        self.load_attributes(cur_dir + '/data/' + data_name + '_attributes')
        #Calcuate input layer size
        self._num_input = 0
        for attr in self.attributes:
            self._num_input += len(attr)
        #Hidden layer has same size with the input layer
        self._num_hidden = self._num_input
        #output layer size
        self._num_output = len(self.classes)
        #Conver all data
        self.training_data = self.convert_all_data(training)
        self.testing_data = self.convert_all_data(testing)
        self.validation_data = self.convert_all_data(validation)
        #Create hidden layer weight matrix
        self._hidden_layer_weights = np.random.uniform(-0.01, 0.01, (self._num_input + 1, self._num_hidden))
        # self._hidden_layer_weights = [[0.1008513], [0.1008513], [0.1]]
        #Create output layer weight matrix
        self._output_layer_weights = np.random.uniform(-0.01, 0.01, (self._num_hidden + 1, self._num_output))
        # self._output_layer_weights = [[0.1343929], [0.1189104]]

    def feed_forward(self, inputvalue):

        self._input_layer = np.insert(inputvalue, 0, 1, axis=1)
        # print self._input_layer
        self._hidden_layer = np.insert(sigmoid(np.dot(self._input_layer, self._hidden_layer_weights)), 0, 1, axis=1)
        self._output_layer = sigmoid(np.dot(self._hidden_layer, self._output_layer_weights))
        # print self._output_layer
        return self._output_layer

    def back_propagate(self, actual_output):
        # print actual_output
        output_layer_error = self.calculate_output_layer_error(actual_output)
        self.update_output_layer_weights(output_layer_error)
        hidden_layer_error = self.calculate_hidden_layer_error(output_layer_error)
        self.update_hidden_layer_weights(hidden_layer_error)
        network_error = np.absolute(output_layer_error).sum() + np.absolute(hidden_layer_error).sum()
        return network_error

    def calculate_output_layer_error(self, actual_output):
        output_layer_error = np.multiply(np.multiply((1 - self._output_layer), self._output_layer),
                                         (actual_output - self._output_layer))
        # print "output_layer_error"
        # print output_layer_error
        return output_layer_error

    def calculate_hidden_layer_error(self, output_layer_error):
        error_sum = np.dot(output_layer_error, np.transpose(self._output_layer_weights))
        # print "error_sum"
        # print error_sum
        hidden_layer_error = np.multiply(np.multiply((1 - self._hidden_layer), self._hidden_layer), error_sum)
        # print "hidden_layer_error"
        # print hidden_layer_error
        return hidden_layer_error

    def update_output_layer_weights(self, output_layer_error):
        delta = np.dot(np.transpose(self._hidden_layer), output_layer_error) * self.learning_rate
        self._output_layer_weights = np.add(self._output_layer_weights, delta)
        # print "_output_layer_weights"
        # print self._output_layer_weights

    def update_hidden_layer_weights(self, hidden_layer_error):
        delta = np.dot(np.transpose(self._input_layer), np.delete(hidden_layer_error, 0, axis=1)) * self.learning_rate
        self._hidden_layer_weights = np.add(self._hidden_layer_weights, delta)
        # print "_hidden_layer_weights"
        # print self._hidden_layer_weights

    #Convert single data to binary inputs and binary output
    def convert_single_data(self, data):
        actual_output = np.zeros(shape=(1, self._num_output))
        index_of_class = self.classes.index(data[0])
        actual_output[0][index_of_class] = 1
        binary_inputs = []
        for index in range(1, len(data)):
            partial_input = np.zeros(shape=(1, len(self.attributes[index - 1])))
            index_of_attr = self.attributes[index - 1].index(data[index])
            partial_input[0][index_of_attr] = 1
            binary_inputs = np.append(binary_inputs, partial_input)
        return binary_inputs, actual_output

    #Convert all data
    def convert_all_data(self, data_set):
        converted_data = []
        for data in data_set:
            converted_data.append(self.convert_single_data(data))
        return converted_data

    def convert_output(self, raw_output):
        output = raw_output[0]
        for index in range(len(output)):
            if output[index] < 0.5:
                output[index] = 0
            else:
                output[index] = 1
        return output

    def train(self):
        training_error_sum = np.inf
        min_validation_error = np.inf
        count = 0
        while training_error_sum > 0.2:
            saved_output_layer_weights = np.copy(self._output_layer_weights)
            saved_output_layer_weights = np.copy(self._output_layer_weights)
            training_error_sum = 0
            count += 1
            for x in range(len(self.training_data)):
                # print x
                self.feed_forward([self.training_data[x][0]])
                training_error_sum += self.back_propagate(self.training_data[x][1])
            # validation_error = self.validation()
            # if validation_error > min_validation_error:
            #     running_time = 20
            # else:
            #     min_validation_error = validation_error
            #     running_time -= 1
            # print validation_error, running_time
            print training_error_sum, count

    def test(self):
        num_total = len(self.testing_data)
        num_pass = 0
        for data in self.testing_data:
            output = self.convert_output(self.feed_forward([data[0]]))
            actual_output = data[1][0]
            if list(output) == list(actual_output):
                num_pass += 1
        print num_pass, num_total

    def validation(self):
        network_error_sum = 0
        for x in range(len(self.validation_data)):
            self.feed_forward([self.training_data[x][0]])
            output_layer_error = self.calculate_output_layer_error(self.training_data[x][1])
            hidden_layer_error = self.calculate_hidden_layer_error(output_layer_error)
            network_error = np.absolute(output_layer_error).sum() + np.absolute(hidden_layer_error).sum()
            network_error_sum += network_error
        return network_error_sum


    """
    Load data
    """

    def load_training_data(self, filename):
        my_file = open(os.path.join(os.path.dirname(__file__), filename))
        lines = my_file.readlines()
        my_file.close()
        num_data = len(lines)  # Number of total data
        training_data = []
        for i in range(num_data):
            training_data.append(lines[i].strip('\n').strip(' ').split(self.separator))
        return training_data

    def load_testing_data(self, filename):
        my_file = open(os.path.join(os.path.dirname(__file__), filename))
        lines = my_file.readlines()
        my_file.close()
        num_data = len(lines)  # Number of total data
        testing_data = []
        for i in range(num_data):
            testing_data.append(lines[i].strip('\n').strip(' ').split(self.separator))
        return testing_data

    def load_validation_data(self, filename):
        my_file = open(os.path.join(os.path.dirname(__file__), filename))
        lines = my_file.readlines()
        my_file.close()
        num_data = len(lines)  # Number of total data
        validation_data = []
        for i in range(num_data):
            validation_data.append(lines[i].strip('\n').strip(' ').split(self.separator))
        return validation_data

    def load_attributes(self, filename):
        my_file = open(filename)
        attributes_raw = my_file.readlines()
        my_file.close()
        self.classes = attributes_raw[0].strip('\n').split(' ')
        for i in range(1, len(attributes_raw)):
            self.attributes.append(attributes_raw[i].strip('\n').split(' '))
        for i in range(0, len(self.attributes)):
            self.attributes_index_list.append(i)

    #Convert single data to binary inputs and binary output
    def convert_single_data(self, data):
        actual_output = np.zeros(shape=(1, self._num_output))
        index_of_class = self.classes.index(data[0])
        actual_output[0][index_of_class] = 1
        binary_inputs = []
        for index in range(1, len(data)):
            partial_input = np.zeros(shape=(1, len(self.attributes[index - 1])))
            index_of_attr = self.attributes[index - 1].index(data[index])
            partial_input[0][index_of_attr] = 1
            binary_inputs = np.append(binary_inputs, partial_input)
        return binary_inputs, actual_output

    #Convert all data
    def convert_all_data(self, data_set):
        converted_data = []
        for data in data_set:
            converted_data.append(self.convert_single_data(data))
        return converted_data


def main():
    data_name = "balance"
    result_id = 5
    ann = ANN()
    ann.learning_rate = 0.7
    ann.setUp(data_name, result_id)
    ann.train()
    ann.test()

    # ann.convert_single_data(['1', '1', '3', '1', '3', '1', '2'])
    # inputvalue = [[0, 1]]
    # ann.feed_forward(inputvalue)
    # ann.back_propagate([[0]])
    # print ann.testing_data
    # print ann.attributes
    # print ann.classes


if __name__ == '__main__':
    main()

# ann = ANN()
# ann.num_input = 2
# ann.num_hidden = 1
# ann.num_output = 1
# ann.learning_rate = 0.3
# ann.setUp()
# inputvalue = [[0, 1]]
# ann.feed_forward(inputvalue)
# ann.back_propagate([[0]])
# print ann.output_layer
# print ann.hidden_layer
# print ann.input_layer
# output_layer_error = ann.calculate_output_layer_error([[1,0]])
# hidden_layer_error = ann.calculate_hidden_layer_error(output_layer_error)
# ann.update_output_layer_weights(output_layer_error)
# ann.update_hidden_layer_weights(hidden_layer_error)
# print output_layer_error
# print hidden_layer_error
