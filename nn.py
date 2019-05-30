'''
Please use Chrome. With other browsers there may be compatibility issues. If for some reason you can not use Chrome, please let the interviewer know. 
'''

'''
Please feel free to use wikipedia, or similar sources or text books to find the descriptions of the Back Propagation algorithm. PLEASE AVOID searching implementations online. Keep the search/research around math/algorithm only, not code/implementation. 

BUNDLED BACKPROPOGATION ALGORITHM:
This is a variation on the Backpropagation algorithm. https://en.wikipedia.org/wiki/Backpropagation . The training function should take in a bundle size (K) and iteration length (N), and instead of updating the weights after going through all the data, it should take a bundle of size K from the data, and update the weights after going through the bundle. It should do this N times. 

Design of the neural net:
There needs to be only one hidden layer, one output layer, and one input layer (the training data). The hidden layer needs to have 3 nodes. Please follow this design. 

THE STEPS:
1) The code should run before you put any new code in. You will see the benchmark F1 score, as well as how the model performs. 
2) We would like to write the remaining code in classify, and train functions. Please keep the function signatures same. 
3) You will only be able to use 'numpy' and 'random' packages.
4) Please comment on your learnings about learning rate, bundle size and iteration length. For what values, what F1 score are you getting? Why?

Time Limit: 90min

'''

import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# each row contains 3 feature values, and a label, total of 4 values. Last value is the label.
np.random.seed(0)

dataset = [
    [0.041, 0.111, 0.214, 0],
    [0.093, 0.166, 0.195, 0],
    [0.106, 0.045, 0.794, 0],
    [1.089, 0.107, 0.452, 1],
    [1.048, 0.112, 0.242, 1],
    [1.044, 1.019, 0.754, 0],
    [1.025, 1.067, 0.557, 0],
    [1.077, 0.994, 0.379, 0],
    [0.13, 1.011, 0.159, 1],
    [0.036, 10.039, 0.46, 1],
    [0.02, 0.03, 0.17, 0],
    [0.03, 0.08, 0.195, 0],
    [0.01, 0.02, 0.769, 0],
    [1, 0.1, 0.396, 1],
    [1.02, 0.05, 0.217, 1],
    [1.02, 0.98, 0.704, 0],
    [0.97, 0.97, 0.552, 0],
    [1.01, 0.93, 0.328, 0],
    [0.04, 1.01, 0.117, 1],
    [0.01, 9.97, 0.365, 1]
]


def F1(dataset, model):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    recall = 0
    precision = 0

    for row in dataset:
        true_label = row[-1]
        if model is None:
            label = 1
        else:
            label = classify(row, model)
        if label == 1:
            if label == true_label:
                true_positive += 1
            else:
                false_positive += 1
        elif label == 0:
            if label == true_label:
                true_negative += 1
            else:
                false_negative += 1

    predicted_positive = true_positive + false_positive
    real_positive = true_positive + false_negative

    if predicted_positive * real_positive == 0:
        return 0

    precision = true_positive / predicted_positive
    recall = true_positive / real_positive
    if precision * recall == 0:
        return 0
    f1score = (precision * recall * 2) / (precision + recall)
    return f1score


def classify(row, model):
    classification = 0
    # INSERT CODE
    X = np.array(row[:-1]).reshape(1, -1)
    w0 = model['w0']
    w1 = model['w1']
    b0 = model['b0']
    b1 = model['b1']
    z0 = np.dot(X, w0) + b0
    a0 = sig(z0)
    z1 = np.dot(a0, w1) + b1
    a1 = sig(z1)
    classification = int(a1>=0.5)
    return classification

def sig(x):
    return 1. / (1. + np.exp(-x))

def dsig(x):
    return x * (1 - x)

def train(dataset, learning_rate, bundle_size, num_iterations):
    # model = {}
    dataset = np.array(dataset)
    batch_size = bundle_size
    
    # INSERT CODE
    for _ in range(num_iterations):
        w0 = model['w0']
        w1 = model['w1']
        b0 = model['b0']
        b1 = model['b1']

        idx = np.random.permutation(list(range(dataset.shape[0])))[:batch_size]
        X = dataset[idx, :-1]
        y = dataset[idx, -1].reshape(bundle_size, -1)

        # forward
        z0 = np.dot(X, w0) + b0
        a0 = sig(z0)
        z1 = np.dot(a0, w1) + b1
        a1 = sig(z1)

        loss = np.sum((a1-y)**2) / batch_size
        losses.append(loss)
        
        # backprop
        dz1 = (a1 - y) * dsig(a1)
        dw1 = np.einsum('nh, nj-> hj', a0, dz1)
        db1 = np.sum(dz1)
        dz0 = np.einsum('nj,hj->nh', dz1, w1) * dsig(a0) # dz1 * W1 * dsig(a1), (n,h)=(n)*(h)*(n,h), NOTE it's W1 not dw1
        db0 = np.sum(dz0, axis=0)
        dw0 = np.einsum('nd,nh->dh', X, dz0)

        model['w0'] -= learning_rate * dw0 / batch_size
        model['w1'] -= learning_rate * dw1 / batch_size
        model['b0'] -= learning_rate * db0 / batch_size
        model['b1'] -= learning_rate * db1 / batch_size

    return model
    

#Hyper params:
BUNDLE_SIZE = 3
LEARNING_RATE = 0.8
NUM_ITERATIONS = 2000
model = {}
nh = 8
model['w0'] = np.random.normal(size=(3, nh))
model['w1'] = np.random.normal(size=(nh, 1))
model['b0'] = np.random.normal(size=(1, nh))
model['b1'] = np.random.random()

# print F1 score of benchmark, without the model
print("The benchmark of dataset is ", F1(dataset, None))
losses = []

# train the model

model = train(dataset, LEARNING_RATE, BUNDLE_SIZE, NUM_ITERATIONS)
print("The model on dataset has F1 score of ", F1(dataset, model))

plt.plot(losses)
plt.show()
