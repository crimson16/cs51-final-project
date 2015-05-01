# cs51-final-project
Recognizing Handwritten Digits Using K-Means Clustering 

By: Olivia Angiuli, Martin Reindl, Ty Rocca, Wilder Wohns 

SETUP INSTRUCTIONS
For users who will want to be able to run our code.

Usage:
cd into the recognize_digits folder

python main_cluster.py {k} {method} {init_type} {prop}

where:
k is the number of clusters
method is “means”, “medians”, or “medoids”
init_type is “random” or “kplusplus”
prop is a number from 0 to 100 with the percentage of training data to train on

Dependencies:
CVXOPT:
    “pip install cvxopt --user”, else follow http://cvxopt.org/install/ 
numpy, scipy, scikit-learn:
    “pip install -U numpy scipy scikit-learn”
Python Image Library (PIL):
    “pip install pil”
django version 1.7:
    “pip install django==1.7”