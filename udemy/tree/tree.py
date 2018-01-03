#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 13:06:03 2017

@author: tan

Entropy and evidence bits: https://stackoverflow.com/questions/1859554/what-is-entropy-and-information-gain
Excellent Guide: http://scikit-learn.org/stable/modules/tree.html#tree

"""

#Applying to Iris Datasets
###########################
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

X = iris.data[:,2:]
y = iris.target

clf = tree.DecisionTreeClassifier(random_state=42)
clf = clf.fit(X,y)

# out of line visualization
from sklearn.tree import export_graphviz
export_graphviz(clf,
                out_file="tree.dot",
                feature_names = iris.feature_names[2:],
                class_names=iris.target_names,
                rounded = True,
                filled = True
                )

# dot -Tpng tree.dot -o tree.png //This converts the file to png

# inline visualization
import graphviz
dot_data = tree.export_graphviz(clf,
                out_file=None,
                feature_names = iris.feature_names[2:],
                class_names=iris.target_names,
                rounded = True,
                filled = True
                )
graph = graphviz.Source(dot_data)
graph
