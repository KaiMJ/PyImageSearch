from os import EX_PROTOCOL
from pickle import NEWOBJ_EX
from nn.perceptron import Perceptron
from nn.neuralnetwork import NeuralNetwork
import numpy as np

# AND
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([[0], [0], [0], [1]])

# def perceptron(X, y):
#     print("[INFO] training perceptron...")
#     p = Perceptron(X.shape[1], alpha=0.1)
#     p.fit(X, y, epochs=20)

#     print("[INFO] testing perceptron...")

#     for (x, target) in zip(X, y):
#         pred = p.predict(x)
#         print("[INFO] data={}, ground-truth={}, pred={}".format(x, target[0], pred))
#     print(p.W)

# perceptron(X, y)

# # OR
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([[0], [1], [1], [1]])

# perceptron(X, y)

# XOR
X = np.array([[0, 0], 
            [0, 1], 
            [1, 0], 
            [1, 1]])
y = np.array([[0], 
            [1], 
            [1], 
            [0]])
print("Shape X and Y: " + str(X.shape) + " " + str(y.shape))

nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y)

for (x, target) in zip(X, y):
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0 else 0
    print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}"
    .format(x, target[0], pred, step))

print(nn.W[0])
print(nn.W[1]) 