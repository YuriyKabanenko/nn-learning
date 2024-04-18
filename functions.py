import numpy as np
from scipy.special import softmax

learning_rate = 0.1


def activation_func(x):
  return 1 / (1 + np.exp(-x))

mapper_sigma = np.vectorize(activation_func)

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
    outputs_1 = mapper_sigma(inputs_1)
    
    inputs_2 = calculate_dot(outputs_1, sl_weights)
    outputs_2 = softmax(inputs_2)
    return outputs_2
    

def calc_weight(cur_weights, output, delta):
    calc_weights = np.empty((0, len(output)))
    delta_counter = 0
    for weight_row in cur_weights:
        new_weight = ((weight_row - output) * delta[delta_counter]) * learning_rate

        calc_weights = np.append(calc_weights, [new_weight], axis=0)    
        delta_counter += 1             
    return calc_weights

    

def train(inputs, expected_predict, fl_weights, sl_weights):
    inputs_1 = calculate_dot(inputs, fl_weights)
    outputs_1 = mapper_sigma(inputs_1)
    
    inputs_2 = calculate_dot(outputs_1, sl_weights)
    outputs_2 = softmax(inputs_2)
    
    actual_predict = np.argmax(outputs_2)
    
    #Second error layer
    sl_error_layer = np.array([outputs_2[actual_predict] - outputs_2[expected_predict]])
    gradient_layer = actual_predict * (1 - actual_predict)
    sl_weights_delta = sl_error_layer * softmax_derivative(inputs_2)
    
    calc_sl_weights = calc_weight(sl_weights, outputs_1, sl_weights_delta)
    
    
    # fl_error_layer = sl_error_layer * sl_weights
    # gradient_layer_1 = outputs_1 * (1 - outputs_1)
    # fl_weights_delta = fl_error_layer * gradient_layer_1

    #First error layer 
    # error_layer_1 = sl_weights_delta * sl_weights
    # gradient_layer_1 = outputs_1 * (1 - outputs_1)
    # fl_weights_delta = error_layer_1 * gradient_layer_1

    #Back propagation
    # calc_fl_weights = fl_weights - (np.dot(inputs.reshape(len(inputs), 1), fl_weights_delta).T * learning_rate)        
    
    return 0
    
    
    