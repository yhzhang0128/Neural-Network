# -*- coding: utf-8 -*- 
# Filename:
#     predict.py
# Description:
#     This file uses NN parameters Theta1 and Theta2 to give 
# prediction for a test trace X

import math
import numpy as np
import numpy.core.fromnumeric as npfunc

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z));

def predict(Theta1, Theta2, X):
    # from Input Layer to Hidden Layer
    X = X.T
    one = np.ones((1,1))
    X1 = np.bmat("one; X")

    # sigmoid of Hidden Layer
    Z2 = Theta1 * X1
    for i in Z2:
        i[0, 0] = sigmoid(i[0, 0])
    A2 = np.bmat("one; Z2")

    # from Hidden Layer to Output Layer    
    Z3 = Theta2 * A2
    ret = list()
    for i in Z3:
        ret.append(sigmoid(i))
        i[0, 0] = sigmoid(i[0, 0])

    # return the Output Layer vector    
    return ret
