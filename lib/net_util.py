"""
a variety of useful functions for building neural nets
"""

import math
import numpy as np
import threading
import multiprocessing
import time
import math

def stochastic_gradient_descent(loss_func, arg_names, loss_func_grad, activation, train_data, \
    num_iterations, input_dim, output_dim, hidden_dim, model, possible_update, reg_params, norm, \
    step_scale, verbose = 0, verbose_n = 1):
    """Runs stochastic gradient descent

    @param loss_func:       full loss based on the whole data set
    @param arg_names:       argument names corresponding to the lossFuncGrad
    @param model:           randomly initialized model
    @param train_data:       list of tuples where each tuple is ([input data], [correct output data])
                            (where the arrays are np arrays)
    @param lossFuncGrad:    a list of functions [dLoss/dW1, dLoss/db1, dLoss/dW2, dLoss/db2]
                            where each function's header is func([input data], [correct output data], model)
    @param activation:      the name of the activation of the middle layer
                            (where each array is an np arrays)
    @param num_iterations:   number of iterations to run gradient descent
    @param input_dim:       dimension of input layer
    @param output_dim:      dimension of output layer
    @param hidden_dim:      dimension of hidden layer
    @param possible_update: possible update to run at each iteration (passed model and the input)
    @param verbose:         0 to NOT print; 1 to print on each update; 2 to print only once per verbose_n iterations
    @param verbose_n:       used only when verbose == 2 (see description of verbose param for details)

    Returns:
        a trained model dictionary
    """

    # Compute average levels

    temp = []
    for elem in zip(*train_data)[0]:
        if elem != None:
            temp.append(elem)
    temp_np = np.array(temp)

    data_len = float(len(temp))
    divisor_vector = np.zeros((output_dim, 1))
    divisor_vector.fill(data_len)

    average_levels = np.sum(temp_np, axis=0)[: output_dim] / divisor_vector

    # Main loop for SGD
    num_updates = 0 # used for step size

    for t in range(num_iterations):

        for input_data, correct_output_data in train_data:

            if input_data == None:
                model['h'] = [np.zeros((hidden_dim, 1))]
                continue

            num_updates += 1
            eta = step_scale / (math.sqrt(num_updates))
            for i in range(len(arg_names)):
                grad = loss_func_grad[i](input_data, correct_output_data, model, average_levels, activation, reg_params, norm)
                model[arg_names[i]] -= eta * grad 

            possible_update(model, input_data)

            if verbose == 1:
                print "Iteration ", t, ": W1 = ", model['W1'], " ; b1 = ", \
                 model['b1'], " ; W2 = ", model['W2'], " ; b2 = ", model['b2']
        if verbose == 1 or verbose == 2 and t % verbose_n == 0:
            print "-------------------------------------------------------"
            print "ITERATION ", t, " COMPLETE"
            #print "W1 = ", model['W1'], " ; b1 = ", model['b1'], " ; W2 = ", \
            # model['W2'], " ; b2 = ", model['b2']
            print "Current Loss: ", loss_func([j[0] for j in train_data], \
                                             [j[1] for j in train_data], \
                                             model, average_levels, activation, \
                                             hidden_dim, possible_update, reg_params, norm)
            
    return (model, loss_func([j[0] for j in train_data], \
                            [j[1] for j in train_data], \
                            model, average_levels, activation, \
                            hidden_dim, possible_update, reg_params, norm))
