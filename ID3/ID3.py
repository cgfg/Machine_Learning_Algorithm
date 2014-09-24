__author__ = 'Xi Chen'
import random
import math
import os
import copy


class ID3:
    def __init__(self, separator):
        # number of data hold
        self.post_pruning_num = 100
        self.test_data_num = 100
        self.training_data = []
        self.testing_data = []
        self.post_pruning_data = []
        self.attributes = []
        self.classes = []
        self.entropy_threshold = 0
        self.separator = separator
        self.attributes_index_list = []

    def load_data(self, filename):
        my_file = open(os.path.join(os.path.dirname(__file__), filename))
        lines = my_file.readlines()
        my_file.close()
        num_data = len(lines)  # Number of total data
        # for i in range(num_data):
        #     self.training_data.append(lines[i].strip('\n').lstrip(' ').split(' '))
        # print("Loaded " + str(num_data) + " training data")
        #Random select hold back data for testing
        test_index = []
        for i in range(self.test_data_num):
            temp = random.randrange(0, num_data)
            while temp in test_index:
                temp = random.randrange(0, num_data)
            test_index.append(temp)
        #split data to one training set and one testing set
        post_pruning_index = []
        for i in range(self.post_pruning_num):
            temp = random.randrange(0, num_data)
            while temp in test_index or temp in post_pruning_index:
                temp = random.randrange(0, num_data)
            post_pruning_index.append(temp)
        for i in range(num_data):
            if i in test_index:
                self.testing_data.append(lines[i].strip('\n').lstrip(' ').split(self.separator))
            elif i in post_pruning_index:
                self.post_pruning_data.append(lines[i].strip('\n').lstrip(' ').split(self.separator))
            else:
                self.training_data.append(lines[i].strip('\n').lstrip(' ').split(self.separator))

    def load_attributes(self, filename):
        my_file = open(filename)
        attributes_raw = my_file.readlines()
        my_file.close()
        self.classes = attributes_raw[0].strip('\n').split(' ')

        for i in range(1, len(attributes_raw)):
            self.attributes.append(attributes_raw[i].strip('\n').split(' '))
        # print("Loaded " + str(len(self.attributes)) + " attributes")
        for i in range(0, len(self.attributes)):
            self.attributes_index_list.append(i)
            # print(self.attributes)

    def create_tree(self, data, attr_val, attribute_index_list):
        #check if we have got the label
        majority_vote = self.vote(data)
        if self.entropy(data) <= self.entropy_threshold or len(attribute_index_list) <= 0:
            label = self.vote(data)
            tree_node = DecisionTreeNode(-1, None, attr_val, label)
            return tree_node
        else:
            #Select best split attribute
            target_attr_index = self.choose_best_attr(data, attribute_index_list)
            #create node
            tree_node = DecisionTreeNode(target_attr_index, None, attr_val, None)
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
                #check if not represented branch
            for val in self.attributes[target_attr_index]:
                if val not in data_map.keys():
                    child = DecisionTreeNode(-1, None, val, majority_vote)
                    tree_node.add_child(child, val)
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
        missing_data = []
        for d in data:
            if d[data_attr_index] not in data_per_branch.keys():
                if d[data_attr_index] != '?':
                    data_per_branch[d[data_attr_index]] = [d]
                else:
                    missing_data.append(d)
            else:
                if d[data_attr_index] != '?':
                    data_per_branch[d[data_attr_index]].append(d)
                else:
                    missing_data.append(d)
        # for key in data_per_branch.keys():
        #     complete_data = data_per_branch.get(key) + missing_data
        #     data_per_branch[key] = complete_data
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
        percent = (num_success / float(total_num)) * 100
        num_fail = total_num - num_success
        print("")
        print("========= Result Before Post-Pruning=========")
        print("Accuracy Rate: %2.2f%%" % percent)
        print("%d success | %d fail | %d total" % (num_success, num_fail, total_num))
        print("Random selected training set size : %d" % len(self.training_data))
        print("Random selected validation set size : %d" % len(self.post_pruning_data))
        print("Random selected testing set size : %d" % len(self.testing_data))
        print("")

    def post_pruning(self, rules_set, data_set):
        for rules in rules_set:
            all_rules = rules[1:]
            num_passed_original = self.test_on_rule(rules, data_set, None)
            for single_rule in copy.copy(all_rules):
                num_passed_after = self.test_on_rule(rules, data_set, single_rule)
                if num_passed_after > num_passed_original:
                    all_rules.remove(single_rule)
        return rules_set


    def test_on_rule(self, rules, data_set, test_remove_node):
        num_passed = 0
        for data in data_set:
            flag = True
            all_rules = rules[1:]
            for single_rule in all_rules:
                if single_rule == test_remove_node:
                    if data[single_rule[0] + 1] != single_rule[1]:
                        flag = False
            if flag is True:
                if rules[0] == data[0]:
                    num_passed += 1
        return num_passed


    def test_on_rules(self, rules_set, data_set):
        num_passed = 0
        for data in data_set:
            for rules in rules_set:
                flag = True
                all_rules = rules[1:]
                for single_rule in all_rules:
                    if data[single_rule[0] + 1] != single_rule[1]:
                        flag = False
                if flag is True:
                    if rules[0] == data[0]:
                        num_passed += 1
        return num_passed

    def run(self):
        root = self.create_tree(self.training_data, None, self.attributes_index_list)
        print("======== Generated Tree ========")
        root.print_tree('')
        print
        rules = root.getRules()
        for rule in rules:
            print rule
        self.test(root)
        self.test_on_rules(rules, self.testing_data)
        new_rules = self.post_pruning(rules, self.post_pruning_data)
        print("======== RULES After Post-Pruning========")
        # for rule in new_rules:
        #     print rule
        num_success = self.test_on_rules(new_rules, self.testing_data)
        total_num = self.test_data_num
        num_fail = total_num - num_success
        percent = num_success * 100 / float(total_num)
        print("Accuracy Rate: %2.2f%%" % percent)
        print("%d success | %d fail | %d total" % (num_success, num_fail, total_num))
        #print len(rules)


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

    def get_attr_index(self):
        return self.attr_index

    def get_class(self, data):
        if self.results:
            return self.results
        elif data[(self.attr_index + 1)] is not '?':
            child = self.children[data[(self.attr_index + 1)]]
            return child.get_class(data)
        else:
            vote_result = {}
            for child in self.children.values():
                result = child.get_class(data)
                if result in vote_result.keys():
                    total = vote_result[result] + 1
                    vote_result[result] = total
                else:
                    vote_result[result] = 1
            max_class = None
            max_vote = float("-inf")
            for key in vote_result.keys():
                if max_vote < vote_result[key]:
                    max_vote = vote_result[key]
                    max_class = key
            return max_class

    def getRules(self):
        if self.results:
            rule = [[self.results, (self.parent_node.get_attr_index(), self.parent_val)]]
            return rule
        elif self.parent_node is None:
            rules = []
            for child in self.children.values():
                child_rules = child.getRules()
                rules += child_rules
            return rules
        else:
            rules = []
            for child in self.children.values():
                child_rules = child.getRules()
                for rule in child_rules:
                    rule += [(self.parent_node.get_attr_index(), self.parent_val)]
                    rules.append(rule)
            return rules

    def get_parent_val(self):
        return self.parent_val

    def get_attr_index(self):
        return self.attr_index

    def get_results(self):
        return self.results

    def print_tree(self, indent=''):
        print(indent + str("\t|\t\t## TEST ATTRIBUTE [" + str(self.attr_index + 1) + "] ##"))
        for child in self.children.values():
            if child.get_results() is None:
                print(indent + "\t" + "\_______IF VALUE IS [" + str(
                    child.get_parent_val()) + "] THEN TEST " + "ATTRIBUTE [" + str(child.get_attr_index() + 1) + "]")
                child.print_tree(indent + "\t")
            else:
                print(indent + "\t\_______" + "IF VALUE IS [" + str(child.get_parent_val()) + "] THEN CLASS is [" + str(
                    child.get_results()) + "]")


def main():
    #Create ID3 object
    id3 = ID3(' ')
    data_name = 'voting_records'
    #Load Data
    cur_dir = os.path.dirname(__file__)  # Get current script file location
    id3.load_data(cur_dir + '/data/' + data_name + '_data')
    # Load attributes
    id3.load_attributes(cur_dir + '/data/' + data_name + '_attributes')
    id3.run()
    # root.print_tree('')


if __name__ == '__main__':
    main()
