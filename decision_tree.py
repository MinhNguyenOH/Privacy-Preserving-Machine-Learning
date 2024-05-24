'''
Helper modules for math and Laplace distribution
'''
import math
import sys
from numpy import random


class Node:
    '''
    Node object represents a node in the decision tree
    '''

    def __init__(self, key):
        '''
        Parameters:
            key: the attribute to test for at this node
        '''
        self.key = key
        self.children = {}


class DecisionTree:
    '''
    Differentially private decision tree class
    '''

    def __init__(self):
        self.root = None
        self.label = None

    def build_tree(self, D, A, label, epsilon, d):
        '''
        Build a decision tree from a given dataset
        Parameters:
            D: input dataset
            A: a set of attributes
            label: the name of label column in the dataset
            epsilon: privacy budget
            d: maximum depth of the decision tree
        '''
        self.label = label
        self.root = self.dp_id3(D, A, epsilon / (2 * (d + 1)), d)

    def dp_id3(self, D, A, epsilon, d):
        '''
        Differentially private ID3 algorithm.
        Used to build the decision tree.
        Parameters:
            D: input dataset
            A: a set of attributes
            epsilon: privacy budget
            d: maximum depth of the decision tree
        Return value:
            the root of the decision tree
        '''
        # total number of rows in the dataset
        N = len(D) + random.laplace(0, 1 / epsilon)
        # maximum number of unique values of an attribute
        if len(A) != 0:
            f = max([D[a].nunique() for a in A])

        # The attribute set is empty or reach final level of the tree
        # or data is to small to split
        if len(A) == 0 or d == 0 or N / (f * D[self.label].nunique()) < math.sqrt(2) / epsilon:
            label_count = {}
            for i in D[self.label].unique():
                label_count[i] = (D[self.label] == i).sum() + \
                    random.laplace(0, 1 / epsilon)
            mode = self.argmax(D[self.label].unique(), label_count)
            return Node(mode)

        G = {}
        for a in A:
            G[a] = self.find_split_entropy(D, a, N, epsilon / (2 * len(A)))
        a_hat = self.argmin(A, G)

        root = Node(a_hat)
        A.remove(a_hat)
        for j in D[a_hat].unique():
            child = self.dp_id3(
                D[D[a_hat] == j], A, epsilon, d - 1)
            root.children[j] = child
        return root

    def find_split_entropy(self, D, a, N, epsilon):
        '''
        Find the minimum split entropy of a given attribute
        Parameters:
            D: input dataset
            a: the attribute to find split entropy for
            N: number of rows in the dataset `D`
            epsilon: privacy budget
        Return value:
            the split entropy of the attribute `a`
        '''
        tot = 0
        for j in D[a].unique():
            subtree_count = (D[a] == j).sum() + random.laplace(0, 1 / epsilon)
            subtree_entropy = 0
            for i in D[self.label].unique():
                predicate = (D[a] == j) & (D[self.label] == i)
                count = predicate.sum() + \
                    random.laplace(0, 1 / epsilon)
                pi = count / subtree_count
                # Non-private implementation doesn't have this `if` statement
                # `pi` must be positive to be used in `log` function
                if pi < 0:
                    continue
                subtree_entropy -= pi * math.log(pi, 2)
            tot += subtree_entropy * subtree_count / N
        return tot

    def predict(self, record):
        '''
        Predict the outcome of a given record
        Parameters:
            record: a record (a row) to predict its outcome
        Return value:
            the value of the label
        '''
        node = self.root
        while len(node.children) != 0:
            node = node.children[record[node.key]]

        return node.key

    def argmin(self, args, values):
        '''
        Helper function to find the argument that gives 
        the minimum value from a list of values.
        Parameters:
            args: list of arguments
            values: list of values associated with each argument
        '''
        if len(args) == 0 or len(values) == 0 or len(args) != len(values):
            print("Invalid argument")
            sys.exit(1)

        min_arg = None
        min_val = float('inf')
        for arg in args:
            if values[arg] < min_val:
                min_arg = arg
                min_val = values[arg]
        return min_arg

    def argmax(self, args, values):
        '''
        Helper function to find the argument that gives 
        the maximum value from a list of values.
        Parameters:
            args: list of arguments
            values: list of values associated with each argument
        '''
        if len(args) == 0 or len(values) == 0 or len(args) != len(values):
            print("Invalid argument")
            sys.exit(1)

        max_arg = None
        max_val = float('-inf')
        for arg in args:
            if values[arg] > max_val:
                max_arg = arg
                max_val = values[arg]
        return max_arg
