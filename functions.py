import numpy as np
from scipy.special import softmax

learning_rate = 0.5


def relu(x):
    return np.maximum(0, x)

mapper_relu = np.vectorize(relu)

def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

def softmax_derivative(z):
    softmax_z = softmax(z)
    
    return softmax_z * (1 - softmax_z)

def MSE(y, Y):
    return np.mean((y - Y) ** 2)

def calculate_dot(inputs, weights):
    dot_list = np.array([])
    for x in weights:
        dot_product = np.dot(inputs, x)
        dot_list = np.append(dot_list, dot_product)  
    return dot_list

def predict(inputs, fl_weights, sl_weights):
    inputs_1 = calculate_dot(inputs, fl_weights)
    outputs_1 = relu(inputs_1)
    
    inputs_2 = calculate_dot(outputs_1, sl_weights)
    outputs_2 = softmax(inputs_2)
    return outputs_2

def calc_weight_output(cur_weights, output, delta):
    calc_weights = np.empty((0, len(output)))
    delta_counter = 0
    for weight_row in cur_weights:
        new_weight = learning_rate * ((weight_row - output) * delta[delta_counter]) 
        calc_weights = np.append(calc_weights, [new_weight], axis=0)    
        delta_counter += 1             
    return calc_weights

def calc_hidden_delta(outputs, sl_weights, sl_delta):
    hidden_delta = np.array([])
    
    for output in outputs:
        row_counter = 0
        weight = sl_weights[:, row_counter]
        delta = relu_derivative(output) * (weight * sl_delta)
        hidden_delta = np.append(hidden_delta, delta.sum())  
    return hidden_delta 
            
def calc_new_weights(delta_weights, old_weights):
    new_weights = []
    for delta, old_weight in zip(delta_weights, old_weights):
        new_weight = old_weight * delta * learning_rate
        new_weights.append(new_weight)   
    return np.array(new_weights) 

def train(inputs, expected_predict, fl_weights, sl_weights):
    expected_list = np.zeros(41)
    expected_list[expected_predict] = 1
    
    inputs_1 = calculate_dot(inputs, fl_weights)
    outputs_1 = mapper_relu(inputs_1)
    
    inputs_2 = calculate_dot(outputs_1, sl_weights)
    outputs_2 = softmax(inputs_2)
    #Second error layer
    sl_error_layer = outputs_2 - expected_list
    sl_weights_delta = sl_error_layer * softmax_derivative(inputs_2)
    #Above is correct
    
    fl_weights_delta = calc_hidden_delta(outputs_1, sl_weights, sl_weights_delta)

    calc_sl_weights = calc_new_weights(sl_weights_delta, sl_weights)
    calc_fl_weights = calc_new_weights(fl_weights_delta, fl_weights)

    return calc_fl_weights, calc_sl_weights
    
    
    