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

EPOCHS = 15
units = len(unique_chars)
input_dims = len(unique_chars)

cell = RNNCell(units, input_dims)
cell.init_layer(0)
optim = SGD()
save = False
loss_history = []

# Training
train_time = time.time()
for epoch in range(EPOCHS):

    epoch_time = time.time()
    eloss = 0
    for namex_idx in track(range(len(names))):
        namex = names[namex_idx]
        X = [None] + [chars_to_index[ch] for ch in namex]
        Y_c = X[1:] + [chars_to_index["<END>"]]

        # transform the input X and label Y into one hot enconding.
        X = one_hot_encoding(X, input_dims)
        Y = np.array(one_hot_encoding(Y_c, units), dtype=int)
        tanhs = []
        softmaxe = Softmax()
        preds = []

        state = np.zeros((units, 1))
        states = [state]

        for char in X:
            y, state, act_in = cell(char)
            # print(y_probs)

            states.append(state)
            preds.append(y)
            tanhs.append(act_in)

        preds = np.array(preds)
        preds = softmaxe(preds)
        loss_val = 0
        losses = [CrossEntropyLoss() for y in preds]
        for pred, loss, y in zip(preds, losses, Y):
            loss_val += loss.forward(pred, y)
        loss_val = loss_val / len(namex)
        eloss += loss_val

        dys = []
        for loss in losses:
            dys.append(loss.backward())
        param_updates = [0, 0, 0, 0, 0]
        ds_prev = np.zeros_like(states[0])

        for dy_idx, dy in reversed(list(enumerate(dys))):
            dx, ds_prev, param_updates = cell.backwards(
                dy,
                ds_prev,
                param_updates,
                X[dy_idx],
                states[dy_idx],
                states[dy_idx + 1],
                tanhs[dy_idx],
            )
        if namex_idx % 10 == 0:
            param_updates = [np.clip(grad, -1, 1, out=grad) for grad in param_updates]
            cell.params, _ = optim.optim(cell.params, param_updates)

    loss_history.append(eloss/len(names))
    if epoch % 1 == 0:
        print(
            f"Epoch {epoch} took {round(time.time()-epoch_time, 3)}s | loss: {eloss/len(names)}"
        )

        print("Names created:", "\n")
        for i in range(4):
            letter = None

            letter_x = np.zeros((input_dims, 1))
            name = []

            while letter != "<END>" and len(name) < 15:
                y, hidden, _ = cell(letter_x)
                y_pred = softmaxe(y.reshape(1, y.shape[0])).reshape(
                    (y.shape[0], y.shape[1])
                )

                index = np.random.choice(indexes, p=y_pred.ravel())
                letter = index_to_chars[index]

                name.append(letter)

                letter_x = np.zeros((input_dims, 1))
                letter_x[index] = 1

            print("".join(name))
print(f"Training took {round(time.time()-train_time)}s, final loss {loss_val}")
from matplotlib import pyplot as plt

plt.plot(loss_history)
plt.show()

generated = []
for i in range(500):
    letter = None
    indexes = list(index_to_chars.keys())

    letter_x = np.zeros((input_dims, 1))
    name = []

    while letter != "<END>" and len(name) < 15:
        y, hidden, _ = cell(letter_x)
        y_pred = softmaxe(y.reshape(1, y.shape[0])).reshape(
            (y.shape[0], y.shape[1])
        )

        index = np.random.choice(indexes, p=y_pred.ravel())
        letter = index_to_chars[index]

        name.append(letter)

        letter_x = np.zeros((input_dims, 1))
        letter_x[index] = 1

    name.pop(-1)
    print("".join(name)) if len(name) > 3 else None
    generated.append("".join(name)) if len(name) > 3 else None
print(f"Non-Empty names: {len(generated)} out of 500")

if save:
    import json
    with open("namegen.json", "w") as f:
        params = {arg: val.tolist() for arg, val in cell.params.items()}
        json.dump(params, f)
