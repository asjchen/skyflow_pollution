"""
Functions common to both neural networks
"""

import math
import numpy as np
import data_util
import test_util
import feed_forward_nn
import elman_rnn
from nn_globals import NetHyperparams
from nn_globals import OUTPUT_DIM, NUM_VARS

def stochastic_gradient_descent(network_setup, train_data, model, \
    verbose=2, verbose_n=1):
    # Compute average levels
    loss_func, loss_func_grad, possible_update, hyper = network_setup
    temp = []
    for elem in zip(*train_data)[0]:
        if elem != None:
            temp.append(elem)
    temp_np = np.array(temp)

    output_dim = model['b2'].shape[0]
    data_len = float(len(temp))
    divisor_vector = np.zeros((output_dim, 1))
    divisor_vector.fill(data_len)

    average_levels = np.sum(temp_np, axis=0)[: output_dim] / divisor_vector

    # Main loop for SGD
    num_updates = 0 # used for step size
    train_inputs = [j[0] for j in train_data]
    train_outputs = [j[1] for j in train_data]
    for t in xrange(hyper.num_iterations):

        for input_data, correct_output_data in train_data:

            if input_data == None:
                model['h'] = [np.zeros((hyper.hidden_dim, 1))]
                continue

            num_updates += 1
            eta = hyper.step_scale / (math.sqrt(num_updates))
            for param in loss_func_grad:
                grad = loss_func_grad[param](input_data, \
                    correct_output_data, model, average_levels, hyper)
                model[param] -= eta * grad 

            possible_update(model, input_data)

            if verbose == 1:
                print "Iteration ", t, ": W1 = ", model['W1'], " ; b1 = ", \
                 model['b1'], " ; W2 = ", model['W2'], " ; b2 = ", model['b2']
        if verbose == 1 or verbose == 2 and t % verbose_n == 0:
            print "-------------------------------------------------------"
            print "ITERATION ", t, " COMPLETE"
            current_loss = loss_func(train_inputs, train_outputs, model, \
                average_levels, possible_update, hyper)
            print "Current Loss: ", current_loss
    final_loss = loss_func(train_inputs, train_outputs, model, \
        average_levels, possible_update, hyper)
    return (model, final_loss)

def run_neural_net(pollution_data_list, hyper, has_feedback):
    if not has_feedback:
        calculate_loss = feed_forward_nn.calculate_loss
        process_data_set = feed_forward_nn.process_data_set
    else:
        calculate_loss = elman_rnn.calculate_loss
        process_data_set = elman_rnn.process_data_set
    (input_vectors, output_vectors) = process_data_set(
        pollution_data_list, hyper.past_scope)

    train_data = zip(input_vectors, output_vectors)
    input_dim = NUM_VARS * hyper.past_scope
    W1 = np.random.randn(hyper.hidden_dim, input_dim) / np.sqrt(input_dim)
    b1 = np.zeros((hyper.hidden_dim, 1))
    W2 = np.random.randn(OUTPUT_DIM, hyper.hidden_dim) / np.sqrt(hyper.hidden_dim)
    b2 = np.zeros((OUTPUT_DIM, 1))
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    if not has_feedback:
        loss_gradients = feed_forward_nn.get_loss_gradients()
        update = feed_forward_nn.none_func
    else:
        loss_gradients = elman_rnn.get_loss_gradients()
        update = elman_rnn.update
        hidden_dim = hyper.hidden_dim
        U = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        h = [np.zeros((hidden_dim, 1))]
        model.update({'U': U, 'h': h})
    network_setup = (calculate_loss, loss_gradients, update, hyper)
    return stochastic_gradient_descent(network_setup, train_data, model)

def test_module(pollution_dirs, hyper, has_feedback):
    pollution_dir_train, pollution_dir_test = pollution_dirs
    if not has_feedback:
        calculate_loss = feed_forward_nn.calculate_loss
        process_data_set = feed_forward_nn.process_data_set
        update = feed_forward_nn.none_func
    else:
        calculate_loss = elman_rnn.calculate_loss
        process_data_set = elman_rnn.process_data_set
        update = elman_rnn.update

    # TRAINING SET
    pollution_data_list_train = data_util.data_from_directory(pollution_dir_train)

    # TEST SET
    pollution_data_list_test = data_util.data_from_directory(pollution_dir_test)
    (model, loss) = run_neural_net(pollution_data_list_train, hyper, has_feedback)

    print 'PROCESSING TRAIN SET'
    (train_inputs, train_outputs) = process_data_set(
        pollution_data_list_train, hyper.past_scope)

    print 'PROCESSING TEST SET'
    (test_inputs, test_outputs) = process_data_set(
        pollution_data_list_test, hyper.past_scope)

    # Calculate average levels
    temp_train = []
    for elem in train_inputs:
        if elem != None:
            temp_train.append(elem)
    temp_np_train = np.array(temp_train)

    data_len_train = float(len(temp_train))
    average_levels_train = np.sum(temp_np_train, axis=0)[: OUTPUT_DIM]
    average_levels_train /= data_len_train

    print "######################## CALCULATING LOSS ########################"
    loss = calculate_loss(train_inputs, train_outputs, model, average_levels_train, \
        update, hyper, print_loss_vector = True)
    print "TRAIN LOSS: ", loss

    (test_inputs, test_outputs) = process_data_set(
        pollution_data_list_test, hyper.past_scope)

    # Calculate average levels for test set
    temp_test = []
    for elem in test_inputs:
        if elem != None:
            temp_test.append(elem)
    temp_np_test = np.array(temp_test)

    data_len_test = float(len(temp_test))
    average_levels_test = np.sum(temp_np_test, axis=0)[: OUTPUT_DIM]
    average_levels_test /= data_len_test

    print "######################## CALCULATING LOSS ########################"
    loss = calculate_loss(test_inputs, test_outputs, model, average_levels_test, \
        update, hyper, print_loss_vector = True)
    print "TEST LOSS: ", loss
    return model

def train_nn(algo, pollution_dirs, hyper, pollutant):
    pollution_dir_train, pollution_dir_test = pollution_dirs
    test_data_set = data_util.data_from_directory(pollution_dir_test)
    print 'READING DATA COMPLETE'
    has_feedback = (algo == 'elman')
    model = test_module(pollution_dirs, hyper, has_feedback)
    scopes = (hyper.past_scope, hyper.future_scope)
    test_util.evaluate_algorithm(scopes, algo, test_data_set, pollutant, \
        hyper.norm, hyper=hyper, model=model)

def parse_nn_input(args):
    args.pollution_dir_test = data_util.remove_slash(args.pollution_dir_test)
    args.pollution_dir_train = data_util.remove_slash(args.pollution_dir_train)
    pollution_dirs = (args.pollution_dir_train, args.pollution_dir_test)
    reg_params = { 'W1': args.reg_w1, 'b1': args.reg_b1, 'W2': args.reg_w2, \
        'b2': args.reg_b2, 'U': args.reg_u }
    hyper = NetHyperparams(args.hidden_dim, args.activation, args.past_scope, \
        reg_params, args.num_iterations, args.future_scope, args.norm, \
        args.step_scale)
    train_nn(args.algo, pollution_dirs, hyper, args.chemical)

