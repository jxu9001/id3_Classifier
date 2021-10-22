# Jerry Xu
# CS 4375 Fall 2021 Homework 1 Part 2 (ID3 Classifier)

import collections
import copy
import math
import random
import sys


# Node class to represent the nodes in our decision tree
class Node:
    def __init__(self, attribute=None, attribute_values=[], label=None,
                 parent_attribute=None, parent_attribute_value=None):
        self.attribute = attribute
        self.attribute_values = attribute_values
        self.label = label
        self.children = {}
        self.parent_attribute = parent_attribute
        self.parent_attribute_value = parent_attribute_value


def get_class_counts(dataset):
    """
    returns a dictionary of the class counts in a dataset
    """
    # key = class in the dataset, value = num examples of that class in the dataset
    class_counts = collections.defaultdict(int)

    for example in dataset:
        class_counts[example["class"]] += 1

    return class_counts


def prior_entropy(dataset):
    """
    calculates the entropy of a dataset prior to splitting
    """
    # key = class in the dataset, value = num examples of that class in the dataset
    class_counts = get_class_counts(dataset)

    # everything is in the same class -> entropy = 0
    if len(class_counts) == 1:
        return 0

    ent = 0
    num_examples = sum(class_counts.values())

    for count in class_counts.values():
        # this effectively makes 0 * log_2(0) = 0
        if count == 0:
            continue
        ent -= (count / num_examples) * math.log2(count / num_examples)

    return ent


def entropy(dataset, attribute, attribute_value):
    """
    calculates the entropy of a dataset with the given value for the given attribute
    e.g. after splitting the RMP reviews dataset on personality,
    calculate the entropy of the boring (personality = 0) professors
    """
    class_counts = collections.defaultdict(int)

    for example in dataset:
        if example[attribute] == attribute_value:
            class_counts[example["class"]] += 1

    if len(class_counts) == 1:
        return 0

    ent = 0
    num_examples = sum(class_counts.values())

    for count in class_counts.values():
        if count == 0:
            continue
        ent -= (count / num_examples) * math.log2(count / num_examples)

    return ent


def information_gain(dataset, attribute):
    """
    calculates the information gain after splitting the dataset on the given attribute
    """
    # key = possible values for the given attribute, value = # of examples with that value
    attribute_value_counts = collections.defaultdict(int)

    for example in dataset:
        attribute_value_counts[example[attribute]] += 1

    ig = prior_entropy(dataset)
    num_examples = sum(attribute_value_counts.values())

    for attribute_value, count in attribute_value_counts.items():
        ig -= (count / num_examples) * entropy(dataset, attribute, attribute_value)

    return ig


def highest_ig_attribute(dataset, remaining_attributes):
    """
    returns the attribute with the highest information gain,
    breaking ties by preferring the earliest one in the list of attributes
    """
    best_attribute = None
    max_gain = -float("inf")

    attributes = remaining_attributes

    for attribute in attributes:
        curr_gain = information_gain(dataset, attribute)
        if curr_gain > max_gain:
            max_gain = curr_gain
            best_attribute = attribute

    return best_attribute


def most_common_class(dataset):
    """
    returns the most common class in the dataset,
    breaking ties by preferring class 0 to class 1 and class 1 to class 2
    """
    class_counts = get_class_counts(dataset)

    return max(sorted(class_counts), key=class_counts.get)


def all_same_class(dataset):
    """
    returns true if all examples in the dataset belong to the same class
    """
    class_values = set()

    for example in dataset:
        class_values.add(example["class"])

    return len(class_values) == 1


def get_all_attribute_values(dataset):
    """
    returns dictionary of all possible values of all attributes in a dataset
    """
    attributes = list(dataset[0].keys())[:-1]
    attribute_values = collections.defaultdict(set)

    for example in dataset:
        for attribute in attributes:
            attribute_values[attribute].add(example[attribute])

    return attribute_values


def is_leaf(dataset, remaining_attributes):
    """
    determines whether a dataset is a leaf node
    if yes, also return what type of leaf node we have in order to apply tiebreaking rules
    """
    # reached leaf node, no examples left
    if len(dataset) == 0:
        return 1

    # reached leaf node, all examples belong to the same class
    if all_same_class(dataset):
        return 2

    # reached leaf node, examples belong to multiple classes
    # in this case, everything about the examples is identical
    # except for the classes they belong to
    if len(remaining_attributes) == 0:
        class_counts = get_class_counts(dataset)
        # now we want to see if there is tie for most frequent class
        if len(set(class_counts.values())) != 1:
            return 3
        else:
            return 4
    else:
        return 0


def id3(dataset, default_label, attribute_values, remaining_attributes, initial_class_counts):
    """
    implementation of the id3 algorithm
    :param dataset: a list of dictionaries, where each dictionary is an example in the dataset
    :param default_label: most common class in the initial dataset
    :param attribute_values: list of all possible values that an attribute in the dataset can take
    :param remaining_attributes: list of attributes that we haven't split on
    :param initial_class_counts: count of examples of each class in the initial dataset
    :return: a decision tree
    """
    # base case: we reached a leaf node
    # now we need to determine which type of leaf node we reached in order to apply the tiebreaking rules
    leaf_type = is_leaf(dataset, remaining_attributes)
    if leaf_type:
        # no examples left, choose class that is most frequent in entire train set
        if leaf_type == 1:
            return Node(label=default_label)

        # all examples belong to same class OR no attributes left
        # so choose most frequent class @ this node
        elif leaf_type == 2 or leaf_type == 3:
            return Node(label=most_common_class(dataset))

        # tie for most frequent class at this node, choose more frequent class in entire train set
        else:
            # count of class values in the dataset and the entire training set
            subset_counts = get_class_counts(dataset)
            class_counts = initial_class_counts

            # key = class values in subset, value = their counts in the entire dataset
            overall_subset_counts = collections.defaultdict(int)

            for class_value in subset_counts:
                overall_subset_counts[class_value] = class_counts[class_value]

            return Node(label=max(sorted(overall_subset_counts), key=overall_subset_counts.get))

    else:
        # find the attribute w/ the highest IG
        max_ig_attribute = highest_ig_attribute(dataset, remaining_attributes)

        # remove the highest IG attribute from the list of attributes so we don't try to split on it later
        remaining_attributes = copy.deepcopy(remaining_attributes)
        remaining_attributes.remove(max_ig_attribute)

        # create a decision tree
        poss_attribute_values = [attribute_value for attribute_value in attribute_values[max_ig_attribute]]
        tree = Node(attribute=max_ig_attribute,
                    attribute_values=poss_attribute_values,
                    label=most_common_class(dataset))

        # split the dataset on the highest IG attribute
        for attribute_value in tree.attribute_values:
            split = [example for example in dataset if attribute_value == example[max_ig_attribute]]

            # recursively create subtrees and link them with the tree
            subtree = id3(split, default_label, attribute_values, remaining_attributes, initial_class_counts)
            subtree.parent_attribute = max_ig_attribute
            subtree.parent_attribute_value = attribute_value
            tree.children[attribute_value] = subtree

        return tree


def display_tree(tree):
    """
    prints the decision tree to screen using iterative preorder traversal
    """
    stack = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        # don't need to print the root node of the decision tree
        if node.parent_attribute:
            parent = node.parent_attribute
            parent_value = str(node.parent_attribute_value)
            label = str(node.label)
            # is it possible to use str.format() for repeated symbols
            line = (depth - 1) * "| " + parent + " = " + parent_value + " : "

            # print non-leaf node
            if node.children:
                print(line.lstrip())

            # print leaf node
            else:
                line += label + " "
                print(line.lstrip())

        for child_node in list(node.children.values())[::-1]:
            stack.append((child_node, depth + 1))


def predict(tree, example):
    """
    returns the tree's prediction when given a new example using iterative DFS
    """
    stack = [tree]
    while stack:
        node = stack.pop()

        # we reached a leaf node, so return its label
        if not node.children:
            return node.label

        # get the attribute of the current node, and the example's corresponding value
        attribute_value = example[node.attribute]

        # go down the branch corresponding to this attribute value
        if attribute_value in node.children:
            stack.append(node.children[attribute_value])


def accuracy(tree, dataset):
    """
    returns the accuracy of the decision tree when tested on a dataset
    """
    correct = 0

    for example in dataset:
        if predict(tree, example) == example["class"]:
            correct += 1

    return correct / len(dataset)


def main():
    # command line args
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]

    # read the training and test sets from their respective files
    with open(train_file_name) as f:
        # some python magic to skip over all empty lines
        train_dataset = [line.split() for line in f if line.strip()]

    attributes = train_dataset.pop(0)
    # more python magic to get the dataset into a usable format
    # now the dataset is a list of dictionaries in the following form: {'a0': 1, 'a1': 1, 'a2': 0, 'a3': 0, 'class': 1}
    train_dataset = [dict(zip(attributes, list(map(int, example)))) for example in train_dataset]

    with open(test_file_name) as f:
        test_dataset = [line.split() for line in f if line.strip()]

    test_dataset.pop(0)
    test_dataset = [dict(zip(attributes, list(map(int, example)))) for example in test_dataset]

    # Part A: build decision tree using the training set and print it to screen
    dataset = train_dataset
    default_label = most_common_class(train_dataset)
    all_attribute_values = get_all_attribute_values(train_dataset)
    remaining_attributes = attributes[:-1]
    initial_class_counts = get_class_counts(train_dataset)
    decision_tree = id3(dataset, default_label, all_attribute_values, remaining_attributes, initial_class_counts)
    display_tree(decision_tree)

    # Part B: Test the tree on the training set and print the accuracy to screen
    train_accuracy = accuracy(decision_tree, train_dataset)
    print("\nAccuracy on training set ({} instances): {}%".format(len(train_dataset), round(100 * train_accuracy, 1)))

    # Part C: Test the tree on the test set and print the result to screen
    test_accuracy = accuracy(decision_tree, test_dataset)
    print("\nAccuracy on test set ({} instances): {}%".format(len(test_dataset), round(100 * test_accuracy, 1)))

    # Part D: re-train the id3 algorithm on train.dat using training set sizes of 100, 200, ..., 800
    # train_set_sizes = list(range(100, 900, 100))
    # results = collections.defaultdict(float)
    #
    # for size in train_set_sizes:
    #     random.shuffle(train_dataset)
    #     decision_tree = id3(train_dataset[:size], default_label, all_attribute_values,
    #                         remaining_attributes, initial_class_counts)
    #     results[size] = accuracy(decision_tree, test_dataset)
    #
    # print("\n")
    # for train_set_size, acc in results.items():
    #     print("Training set size = {}, Accuracy on test set = {}".format(train_set_size, acc))


if __name__ == "__main__":
    main()
