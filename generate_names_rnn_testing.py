import time
import numpy as np
from matplotlib import pyplot as plt
import random

from rich.progress import track

from fwuffy.lunak.utils import one_hot_encoding
from fwuffy.lunak.layers import RNNCell
from fwuffy.lunak.activations import Softmax, Tanh
from fwuffy.lunak.losses import CrossEntropyLoss
from SGD import SGD

class CrossEntropyLoss:
    def __init__(self):
        self.type = 'CELoss'
        self.eps = 1e-15
        self.softmax = Softmax()
    
    def forward(self, Y_hat, Y):
        """
        Computes the forward propagation.
        Parameters
        ----------
        Y_hat : numpy.array
            Array containing the predictions.
        Y : numpy.array
            Array with the real labels.
        
        Returns
        -------
        Numpy.arry containing the cost.
        """
        self.Y = Y
        self.Y_hat = Y_hat

        _loss = - Y * np.log(self.Y_hat)
        loss = np.sum(_loss, axis=0).mean()

        return np.squeeze(loss) 

    def backward(self):
        """
        Computes the backward propagation.
        Returns
        -------
        grad : numpy.array
            Array containg the gradients of the weights.
        """
        grad = self.Y_hat - self.Y
        
        return grad

with open("./person_names.txt", "r") as f:
    names_str = f.read()
    f.seek(0)
    names = f.readlines()

# character mappings
unique_chars = sorted(list(set(names_str.lower())))

chars_to_index = {char: idx for idx, char in enumerate(unique_chars)}
index_to_chars = {idx: char for idx, char in enumerate(unique_chars)}
index_to_chars[0] = "<END>"
chars_to_index.pop("\n")
chars_to_index["<END>"] = 0

names = [name.strip().lower() for name in names]
random.shuffle(names)
print(names[:5])

EPOCHS = 100001
units = len(unique_chars)
input_dims = len(unique_chars)

cell = RNNCell(units, input_dims)
cell.init_layer(0)
optim = SGD(0.01)
loss_history = []

# Training
for epoch in range(EPOCHS):
    
    # create the X inputs and Y labels
    index = epoch % len(names)
    X = [None] + [chars_to_index[ch] for ch in names[index]] 
    Y_c = X[1:] + [chars_to_index["<END>"]]

    # transform the input X and label Y into one hot enconding.
    X = one_hot_encoding(X, input_dims)
    Y = np.array(one_hot_encoding(Y_c, units), dtype=int)
    tanhs = [Tanh() for c in X]
    softmaxe = Softmax()
    preds = []
    
    
    state = np.zeros((units, 1))
    states = [state]
    
    for char, tanh in zip(X, tanhs):
        y, state = cell(char, state, tanh)
        y_probs = softmaxe(y)
        #print(y_probs)
        
        states.append(state)
        preds.append(y_probs)
    
    preds = np.array(preds)
    loss_val = 0
    losses = [CrossEntropyLoss() for y in preds]
    for pred, loss, y in zip(preds, losses, Y):
        loss_val += loss.forward(pred, y)
    if epoch % 1000 == 0:
        loss_history.append(loss_val/len(preds))
    
    dys = []
    for loss in losses:
        dys.append(loss.backward())
    param_updates = [0,0,0,0,0]
    ds_prev = np.zeros_like(states[0])
    
    for dy_idx, dy in reversed(list(enumerate(dys))):
        dx, ds_prev, param_updates = cell.backwards(dy, ds_prev, param_updates, X[dy_idx], states[dy_idx], states[dy_idx+1], tanhs[dy_idx])
    
    #param_updates = [p / len(X) for p in param_updates]
    cell.params, _ = optim.optim(cell.params, param_updates)
    
    if epoch % 10000 == 0:
        print ("Loss after epoch %d: %f" % (epoch, loss_val/len(preds)))

        print('Names created:', '\n')
        for i in range(4):
            letter = None
            indexes = list(index_to_chars.keys())

            letter_x = np.zeros((input_dims, 1))
            name = []

            # similar to forward propagation.
            layer_tanh = Tanh()
            hidden = np.zeros((units , 1))

            while letter != '<END>' and len(name)<15:
                input_softmax, hidden = cell(letter_x, hidden, layer_tanh)
                y_pred = softmaxe(input_softmax)

                index = np.random.choice(indexes, p=y_pred.ravel())
                letter = index_to_chars[index]

                name.append(letter)

                letter_x = np.zeros((input_dims, 1))
                letter_x[index] = 1

            print("".join(name))
from matplotlib import pyplot as plt
plt.plot(loss_history)
plt.show()