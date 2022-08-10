import time
import numpy as np
from matplotlib import pyplot as plt
import random

from rich.progress import track

from fwuffy.lunak.utils import one_hot_encoding
from fwuffy.lunak.layers import RNN, Embedding, LSTMCell, Linear
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

names = [name.strip().lower() if len(name) < 10 else name.strip().lower()[:9] for name in names]
random.shuffle(names)
print(names[:5])
tokens = [[None]+[chars_to_index[char] for char in name]+[0]*(9-len(name)) for name in names]
one_hot_encoded = np.array([one_hot_encoding(tok, len(unique_chars)) for tok in tokens])
batches = np.array(np.split(one_hot_encoded, 61))
tokensy = [[chars_to_index[char] for char in name]+[0]*(10-len(name)) for name in names]
one_hot_encodedy = np.array([one_hot_encoding(tok, len(unique_chars)) for tok in tokensy], dtype=int)
batchesy = np.array(np.split(one_hot_encodedy, 61))

EPOCHS = 1
units = 27
input_dims = len(unique_chars)
embedding_layer0 = Embedding(32, 27)
embedding_layer0.in_dims = batches.shape[2:]
embedding_layer0.init_layer(0)
rnn_layer0 = RNN(LSTMCell(units))
linear0 = Linear(27, activation="softmax")
linear0.in_dims = 27
linear0.init_layer(0)
print(linear0.out_dims)
print(embedding_layer0.out_dims)
rnn_layer0.in_dims = embedding_layer0.out_dims
rnn_layer0.init_layer(0)

for i in range(EPOCHS):
    
    for batch, batchy in zip(track(batches), batchesy):
        x=embedding_layer0(batch)
        x=rnn_layer0(x)
        x=linear0(x.reshape((-1, units))).reshape((-1, 10, 27, 1))
        
        losses = [[CrossEntropyLoss() for el in seq] for seq in x]
        loss_v = 0
        for seq_idx, seq in enumerate(x):
            seqloss = 0
            for el_idx, el in enumerate(seq):
                seqloss += losses[seq_idx][el_idx].forward(el, batchy[seq_idx][el_idx])
            loss_v += seqloss / len(seq)
        
        dys = []
        for seq_idx, seq in enumerate(x):
            seqdy = []
            for el_idx, el in enumerate(seq):
                seqdy.append(losses[seq_idx][el_idx].backward())
            dys.append(seqdy)
        dy = np.array(dys)
        
        dy = linear0.backwards(dy.reshape((-1, units))).reshape((-1, 10, 27, 1))
        dy = rnn_layer0.backwards(dy)
        dy = embedding_layer0.backwards(dy)
        embedding_layer0._update_params(0.01)
        rnn_layer0._update_params(0.01)
        linear0._update_params(0.01)
    print(loss_v/len(x))
    for i in range(1):
        letter = None

        letter_x = np.zeros((1, 1, input_dims, 1))
        name = []

        while letter != "<END>" and len(name) < 10:
            x=embedding_layer0(letter_x)
            x=rnn_layer0(x)
            x=linear0(x.reshape((-1, units))).reshape((1, 1, 27, 1))

            index = np.random.choice(indexes, p=x.ravel())
            letter = index_to_chars[index]
            name.append(letter)

            letter_x = np.zeros((1, 1, input_dims, 1))
            letter_x[0,0,index] = [1]

        print("".join(name))
generated=[]
for i in range(100):
    letter = None

    letter_x = np.zeros((1, 1, input_dims, 1))
    name = []

    while letter != "<END>" and len(name) < 10:
        x=embedding_layer0(letter_x)
        x=rnn_layer0(x)
        x=linear0(x.reshape((-1, units))).reshape((1, 1, 27, 1))

        index = np.random.choice(indexes, p=x.ravel())
        letter = index_to_chars[index]
        name.append(letter)

        letter_x = np.zeros((1, 1, input_dims, 1))
        letter_x[0,0,index] = [1]

    name.pop(-1)
    print("".join(name)) if len(name) > 2 else None
    generated.append("".join(name)) if len(name) > 2 else None
print(f"Non-Empty names: {len(generated)} out of 100")
lengths = [len(a) for a in generated]
plt.hist(lengths)
plt.show()