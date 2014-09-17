__author__ = 'Xi Chen'
import random
import math
import os
import copy

class ID3:

    def __init__(self):
        # number of data hold
        self.post_pruning_num = 30
        self.test_data_num = 60
        self.training_data = []
        self.testing_data = []
        self.attributes = []
        self.classes = []
        self.entropy_threshold = 0.01

    def load_data(self, filename):
        my_file = open(os.path.join(os.path.dirname(__file__), filename))
        lines = my_file.readlines()
        my_file.close()
        num_data = len(lines) # Number of total data
        # for i in range(num_data):
        #     self.training_data.append(lines[i].strip('\n').lstrip(' ').split(' '))
        print("Loaded " + str(num_data) + " training data")
        #Random select hold back data for testing
        test_index = []
        for i in range(self.test_data_num):
            temp = random.randrange(0, num_data)
            while temp in test_index:
                temp = random.randrange(0, num_data)
            test_index.append(temp)
        #split data to one training set and one testing set

        for i in range(num_data):
            if i in test_index:
                self.testing_data.append(lines[i].strip('\n').lstrip(' ').split(' '))
            else:
                self.training_data.append(lines[i].strip('\n').lstrip(' ').split(' '))

    def load_attributes(self, filename):
        my_file = open(filename)
        attributes_raw = my_file.readlines()
        my_file.close()
        self.classes = attributes_raw[0].strip('\n').split(',')

        for i in range(1, len(attributes_raw)):
            self.attributes.append(attributes_raw[i].strip('\n').split(','))
        print("Loaded " + str(len(self.attributes)) + " attributes")
        print(self.attributes)

    def create_tree(self, data, attr_val, attribute_index_list):
        #check if we have got the label
        current_entropy = self.entropy(data)
        if current_entropy <= self.entropy_threshold or len(attribute_index_list) == 0:
            label = self.vote(data)
            tree_node = DecisionTreeNode(-1, None, attr_val, label)
            # print("label : " + str(label))
            return tree_node
        else:
             #Select best split attribute
            target_attr_index = self.choose_best_attr(data, attribute_index_list)
            #create node
            tree_node = DecisionTreeNode(target_attr_index, None, None, None)
             # copy current attribute_index_list, then remove the used one
            children_attribute_index_list = copy.deepcopy(attribute_index_list)
            children_attribute_index_list.remove(target_attr_index)
            # split data use the selected attribute
            data_map = self.split_data(data, target_attr_index)
            for val in data_map.keys():
                child = self.create_tree(data_map[val], val, copy.deepcopy(children_attribute_index_list))
                tree_node.add_child(child, val)
                # print("if : val = " + str(val))
                child.set_parent(tree_node)
            return tree_node

    def choose_best_attr(self, data, attribute_index_list):
        min_entropy = float("inf")
        min_index = -1
        for attr_index in attribute_index_list:
            entropy = self.impurity(data, attr_index)
            if entropy < min_entropy:
                min_entropy = entropy
                min_index = attr_index
        return min_index

    def impurity(self, data, target_attr_index):
        total_num = float(len(data))
        data_per_branch = self.split_data(data, target_attr_index)
        total_impurity = 0
        for branch in data_per_branch.keys():
            branch_data = data_per_branch[branch]
            data_num = float(len(branch_data))
            total_impurity += (data_num * self.entropy(branch_data) / float(total_num))
        return total_impurity

    def entropy(self, data):
        data_num = float(len(data))
        num_per_class = {}
        for d in data:
            if d[0] in num_per_class.keys():
                num_per_class[d[0]] += 1
            else:
                num_per_class[d[0]] = 1
        #calculate entropy
        entropy = 0
        for num in num_per_class.values():
            entropy -= ((num / float(data_num)) * (math.log(num / float(data_num)) / math.log(2)))
        return entropy

    def split_data(self, data, target_attr_index):
        data_attr_index = target_attr_index + 1
        data_per_branch = {}
        for d in data:
            if d[data_attr_index] not in data_per_branch.keys():
                data_per_branch[d[data_attr_index]] = [d]
            else:
                data_per_branch[d[data_attr_index]].append(d)
        return data_per_branch

    def vote(self, data):
        num_per_class = {}
        for d in data:
            if d[0] in num_per_class.keys():
                num_per_class[d[0]] += 1
            else:
                num_per_class[d[0]] = 1
        majority_class = None
        max_votes = float("-inf")
        for c in num_per_class.keys():
            if num_per_class[c] > max_votes:
                majority_class = c
                max_votes = num_per_class[c]
        return majority_class

    def test(self, root_node):
        num_success = 0
        total_num = len(self.testing_data)
        for d in self.testing_data:
            if d[0] == root_node.get_class(d):
                num_success += 1
        print(num_success)
        print(total_num)


class DecisionTreeNode:
    def __init__(self, attr_index=-1, parent_node=None, parent_val=None, results=None):
        self.attr_index = attr_index
        self.parent_node = parent_node
        self.parent_val = parent_val
        self.results = results
        self.children = {}

    def add_child(self, child_node, child_val):
        self.children[child_val] = child_node

    def set_parent(self, parent):
        self.parent_node = parent

    def get_class(self,data):
        if self.results:
            return self.results
        else:
            if data[(self.attr_index + 1)] in self.children.keys():
                child = self.children[data[(self.attr_index + 1)]]
                return child.get_class(data)
            else:
                #Problem here
                return self.children.keys()[0]



def main():
    #Create ID3 object
    id3 = ID3()
     #Load Data
    cur_dir = os.path.dirname(__file__) # Get current script file location
    id3.load_data(cur_dir + '/data/monk1_data')
    # Load attributes
    id3.load_attributes(cur_dir + '/data/monk1_attributes')
    root = id3.create_tree(id3.training_data, None, [0, 1, 2, 3, 4, 5])
    id3.test(root)
if __name__ == '__main__':
    main()
