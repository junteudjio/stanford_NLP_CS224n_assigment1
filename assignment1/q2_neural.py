#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z1 = np.dot(data, W1) + b1
    h = sigmoid(z1)
    logits = np.dot(h, W2) + b2
    probas = softmax(logits)
    cost = - np.sum(np.log(probas) * labels)
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation

    ### 1. Last layer
    # error at the last layer
    gradcost = probas - labels # or gradz2
    # update neurons at last layer using input and error of last layer
    gradW2 = np.dot(h.T, gradcost) # H x Dy
    gradb2 = np.sum(gradcost, axis=0).reshape(1,-1) # 1 x Dy

    ### 2. Hidden layer
    #compute error at hidden layer using gradient comming from last layer
    gradh = gradcost.dot(W2.T)
    #IMPORTANT: here we the element-wise multiplication because when deriving the composition sigmoid(z1=h.W2 + b2),
    #sigmoid() is consider as a single variable function of z1.
    gradz1 = gradh * sigmoid_grad(h)

    # update the neurons of the hidden layer using input and error of hidden layer
    gradW1 = np.dot(data.T, gradz1) # Dx x 1
    gradb1 = np.sum(gradz1, axis=0).reshape(1,-1) # 1 x H
    # ### END YOUR CODE
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()
