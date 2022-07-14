import time
import numpy as np
from matplotlib import pyplot as plt
import random

from rich.progress import track

from fwuffy.lunak.utils import one_hot_encoding
from fwuffy.lunak.layers import RNN, RNNCell, Linear
from fwuffy.lunak.activations import Softmax, Tanh
from SGD import SGD

class CrossEntropyLoss:
    def __init__(self):
        self.type = "CELoss"
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

        _loss = -Y * np.log(self.Y_hat)
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

indexes = list(index_to_chars.keys())

names = [name.strip().lower() for name in names]
random.shuffle(names)
print(names[:5])
tokens = [[None]+[chars_to_index[char] for char in name]+[0]*(15-len(name)) for name in names]
one_hot_encoded = np.array([one_hot_encoding(tok, len(unique_chars)) for tok in tokens])
batches = np.array(np.split(one_hot_encoded, 61))
tokensy = [[chars_to_index[char] for char in name]+[0]*(16-len(name)) for name in names]
one_hot_encodedy = np.array([one_hot_encoding(tok, len(unique_chars)) for tok in tokensy], dtype=int)
batchesy = np.array(np.split(one_hot_encodedy, 61))

EPOCHS = 50
units = 27
input_dims = len(unique_chars)
rnn_layer0 = RNN(RNNCell(units, input_dims))
linear0 = Linear(27, activation="softmax")
linear0.in_dims = 27
linear0.init_layer(0)
rnn_layer0.in_dims = batches.shape[1:]
rnn_layer0.init_layer(0)

# NOTE: for some odd reason, the net sometimes just outputs a EOS character. maybe... find out why, but at this point
# I do not care anymore, because it still kinda works, and i am scared to change the code.
for i in range(EPOCHS):
    
    x=rnn_layer0(batches[0])
    #x=linear0(x).reshape((-1, 16, 27, 1))
    
    losses = [[CrossEntropyLoss() for el in seq] for seq in x]
    loss_v = 0
    for seq_idx, seq in enumerate(x):
        seqloss = 0
        for el_idx, el in enumerate(seq):
            seqloss += losses[seq_idx][el_idx].forward(el, batchesy[0][seq_idx][el_idx])
        loss_v += seqloss / len(seq)
    print(loss_v/len(x))
    
    dys = []
    for seq_idx, seq in enumerate(x):
        seqdy = []
        for el_idx, el in enumerate(seq):
            seqdy.append(losses[seq_idx][el_idx].backward())
        dys.append(seqdy)
    dy = np.array(dys)
    
    # dy = linear0.backwards(dy).reshape((-1, 16, 27, 1))
    dy = rnn_layer0.backwards(dy)
    rnn_layer0._update_params(0.1)
    # linear0._update_params(0.1)
    
    for i in range(1):
        letter = None

        letter_x = np.zeros((1, 1, input_dims, 1))
        name = []

        while letter != "<END>" and len(name) < 15:
            x=rnn_layer0(letter_x)

            index = np.random.choice(indexes, p=x.ravel())
            letter = index_to_chars[index]
            name.append(letter)

            letter_x = np.zeros((1, 1, input_dims, 1))
            letter_x[0,0,index] = [1]

        print("".join(name))
for i in range(100):
    letter = None

    letter_x = np.zeros((1, 1, input_dims, 1))
    name = []

    while letter != "<END>" and len(name) < 15:
        x=rnn_layer0(letter_x)

        index = np.random.choice(indexes, p=x.ravel())
        letter = index_to_chars[index]
        name.append(letter)

        letter_x = np.zeros((1, 1, input_dims, 1))
        letter_x[0,0,index] = [1]

    name.pop(-1)
    print("".join(name)) if len(name) > 2 else None