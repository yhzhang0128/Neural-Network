# -*- coding: utf-8 -*- 
# Filename:
#     learning.py
# Description:
#     This file contains all functions for training a Neural Network

import os
import numpy as np
import scipy.io as spio
import numpy.core.fromnumeric as npfunc

# Sigmoid function for a vector
#    sigmoid(x) = 1 / (1 + e^(-x))
def sigmoid(z):
    x, y = npfunc.shape(z)
    ret = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            ret[i,j] = 1.0 / (1.0 + np.exp(-z[i,j]))
    return ret

# Sigmoid Gradient for a peticular number
def sigmoidGradient(z):
    term1 = 1.0 / (1.0 + np.exp(-z))
    return term1 * (1-term1)

# Log Function for a vector
def eachlog(z):
    ret = []
    for i in z:
        ret.append(np.log(i))
    ret = np.array(ret)
    return ret

# Function of cost and gradient for fmin_l_bfgs_b
# def nnCostFunction(Theta,
#                   input_layer_size,    
#                   hidden_layer_size,   
#                   num_labels,          
#                   X, y, lambda1)
#
#       Theta:  a Combination vector of Theta1 and Theta2
#       input_layer_size:  in this case is 400(20*20)
#       hidden_layer_size:  in this case I choose 25
#       num_laybels:  output layer size which is 10 (10 numbers)
#       X:  training example vector
#       y:  correct answer for training example
#       lambda:  parameter for regularization
#
#       return a tuple (J, grad(1:n))
#           where J is the cost with parameter Theta
#           and   grad is the gradient vector of Theta       

def nnCostFunction(Theta,
                   input_layer_size,    
                   hidden_layer_size,   
                   num_labels,          
                   X, y, lambda1):
    # Extract Theta1 and Theta2 from Theta
    Theta1_sz = hidden_layer_size * (input_layer_size + 1)
    Theta1 = Theta[:Theta1_sz].reshape(input_layer_size+1, hidden_layer_size)
    Theta1 = Theta1.T
    Theta2_sz = num_labels * (hidden_layer_size + 1)
    Theta2 = Theta[Theta1_sz:].reshape(hidden_layer_size + 1, num_labels)
    Theta2 = Theta2.T

    # There are m samples    
    m = npfunc.shape(X)[0]
    
    # Objects to return   (Cost and Gradient)
    J = 0
    Theta1_grad = np.zeros(npfunc.shape(Theta1))
    Theta2_grad = np.zeros(npfunc.shape(Theta2))

    # Add Constant term into equation
    tmp = [[1] for i in range(m)]
    X = np.bmat('tmp X')

    # Count Cost J
    a2 = np.copy(sigmoid(Theta1 * X.T))    
    a2m = npfunc.shape(a2)[1]
    tmp = np.ones((1, a2m))
    a2 = np.bmat('tmp ; a2')
    a2 = a2.T

    h_theta = np.copy(sigmoid((Theta2 * a2.T).T))
    
    
    for i in range(m):
        yi = np.zeros((1, num_labels))
        yi[0,y[i,0]] = 1
        J = J - np.inner(yi, eachlog(h_theta[i]))[0] - np.inner(1-yi, eachlog(1-h_theta[i]))[0]

    J = J / m

    # Count Gradient of Theta1 and Theta2
    # for algorithm Backpropagation
    for t in range(m):
        a1 = X[t].T
        z2 = Theta1 * a1
        a2 = np.copy(sigmoid(z2))

        tmp = np.mat([[1]])
        a2 = np.bmat('tmp ; a2')
        z3 = Theta2 * a2
        a3 = np.copy(sigmoid(z3))

        yt = np.zeros((num_labels, 1))
        yt[y[t,0],0] = 1
        delta3 = np.mat(a3 - yt)


        delta2 = Theta2.T * delta3
        tmp = npfunc.shape(delta2)[0]
        for i in range(tmp):
            if (i == 0):
                delta2[i, 0] = 0
            else:
                delta2[i, 0] = delta2[i,0] * sigmoidGradient(z2[i-1,0])

        Theta2_grad = Theta2_grad + delta3 * a2.T
        Theta1_grad = Theta1_grad + delta2[1:] * a1.T

        
    Theta1_grad = Theta1_grad / m
    Theta2_grad = Theta2_grad / m
    
    
    # Normalization
    reg = 0
    # Theta1
    l = npfunc.shape(Theta1)[0]
    for i in range(l):
        tmp1 = np.array(Theta1[i])
        reg = reg + np.inner(tmp1[1:], tmp1[1:])        

        reg_Theta = Theta1[i]
        reg_Theta[0] = 0
        Theta1_grad[i] = Theta1_grad[i] + reg_Theta * lambda1 / m
    # Theta2
    l = npfunc.shape(Theta2)[0]
    for i in range(l):
        tmp1 = np.array(Theta2[i])
        reg = reg + np.inner(tmp1[1:], tmp1[1:])

        reg_Theta = Theta2[i]
        reg_Theta[0] = 0
        Theta2_grad[i] = Theta2_grad[i] + reg_Theta * lambda1 / m

    J = J + reg * lambda1 / (2*m)

    # Combine Theta1_grad and Theta2_grad into grad
    grad = []
    for j in range(input_layer_size + 1):
        for i in range(hidden_layer_size):
            grad.append(Theta1_grad[i,j])

    for j in range(hidden_layer_size + 1):
        for i in range(num_labels):
            grad.append(Theta2_grad[i,j])    
    grad = np.array(grad)

    print "Cost: ", J
    return (J, grad)


# Training Function
# This function train the NN with parameters in file trace
# and output the result into file dest. Both trace and dest
# is .mat format which also adapts Matlab    
def train(trace, dest):    
    import scipy.io as spio
    raw = spio.loadmat(trace)
    
    # Optimize function parameters
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    X = np.mat(raw['X'])
    y = np.mat(raw['y'])
    lambda1 = 1
    
    # Randomly initialize Theta1 and Theta2
    from random import random
    total = (hidden_layer_size * (input_layer_size+1)) + (num_labels * (hidden_layer_size + 1))
    Theta = []
    eposilon_init = 0.12
    for i in range(total):
        Theta.append(random() * eposilon_init*2 - eposilon_init)
    Theta = np.array(Theta)
    
    # Use BFGS algorithm to optimize the NN parameters
    # BFGS is provided in scipy
    from scipy.optimize import fmin_l_bfgs_b
    result = fmin_l_bfgs_b(nnCostFunction, Theta, None, (input_layer_size, hidden_layer_size, num_labels, X, y, lambda1))

    # Extract Theta1 and Theta2 form result
    Theta = result[0]
    Theta1_sz = hidden_layer_size * (input_layer_size + 1)
    Theta1 = Theta[:Theta1_sz].reshape(input_layer_size+1, hidden_layer_size)
    Theta1 = Theta1.T
    Theta2_sz = num_labels * (hidden_layer_size + 1)
    Theta2 = Theta[Theta1_sz:].reshape(hidden_layer_size + 1, num_labels)
    Theta2 = Theta2.T

    # Save Theta1 and Theta2 into dest (a .mat file)
    total = {'Theta1':Theta1, 'Theta2':Theta2}
    spio.savemat(dest, total)

