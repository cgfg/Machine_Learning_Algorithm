import os
import sys


class NaiveBayes:
    def __init__(self, weight_m, data_name, result_id):
        self.weight_m = weight_m
        self.counts = {}
        self.data_name = data_name
        self.attributes = []
        self.classes = []
        # self.attributes_index_list =[]
        self.separator = ' '
        cur_dir = os.path.dirname(__file__)  # Get current script file location
        # sys.stdout = open(cur_dir + '/data/' + data_name + '_ANN_RESULTS_' + str(result_id) + '.txt', 'w')
        cur_dir = os.path.dirname(__file__)  # Get current script file location
        self.testing = self.load_testing_data(
            cur_dir + '/data/' + data_name + '_results_testing_data_' + str(result_id))
        self.training = self.load_training_data(
            cur_dir + '/data/' + data_name + '_results_training_data_' + str(result_id))
        self.validation = self.load_validation_data(
            cur_dir + '/data/' + data_name + '_results_pruning_data_' + str(result_id))
        # Load attributes
        self.load_attributes(cur_dir + '/data/' + data_name + '_attributes')
        for c in self.classes:
            attr_map = {}
            for attr_index in range(len(self.attributes)):
                attr_map[attr_index] = {}
                for attr_val in self.attributes[attr_index]:
                    attr_map[attr_index][attr_val] = 0
            self.counts[c] = [0, attr_map]
        self.learn()

    def learn(self):
        for data in self.training:
            data_class = data[0]
            data_attr = data[1:]
            self.counts[data_class][0] += 1
            for attr_index in range(len(data_attr)):
                self.counts[data_class][1][attr_index][data_attr[attr_index]] += 1

    def calculate_possibility(self, data, class_name):
        all_count = 0
        for value in self.counts.values():
            all_count += value[0]
        possibility = self.counts[class_name][0] / float(all_count)
        for attr_index in range(len(data)):
            prior_estimate = 1 / float(len(self.attributes[attr_index]))
            possibility *= (
            (self.counts[class_name][1][attr_index][data[attr_index]] + self.weight_m * prior_estimate) / float(
                self.counts[class_name][0] + self.weight_m))
        return possibility

    def classify(self, data):
        max_possibility = float("-inf")
        max_class = None
        for class_name in self.classes:
            possibility = self.calculate_possibility(data, class_name)
            if possibility > max_possibility:
                max_possibility = possibility
                max_class = class_name
        return max_class

    def test_all(self):
        correct_count = 0
        all_count = 0
        for data in self.testing:
            all_count += 1
            real_class = data[0]
            predict_class = self.classify(data[1:])
            if real_class == predict_class:
                correct_count += 1
        return correct_count / float(all_count)


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
            # for i in range(0, len(self.attributes)):
            #     self.attributes_index_list.append(i)


def main(argv):
    weight_m = 10
    data_name = "voting"
    result_id = 4

    nb = NaiveBayes(weight_m, data_name, result_id)
    print nb.test_all()


if __name__ == '__main__':
    main(sys.argv[1:])

