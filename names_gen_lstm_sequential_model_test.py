from fwuffy.lunak.layers import LSTMCell, RNN, Linear, Embedding, Reshape
from fwuffy.lunak.nn import Sequential, load_model
from fwuffy.lunak.losses import CrossEntropyLoss
from fwuffy.lunak.utils import one_hot_encoding
from fwuffy.lunak.checkpoints import Checkpointer

import random
import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track

# Prepare training data
with open("./person_names.txt", "r") as f:
    names_str = f.read()
    f.seek(0)
    names = f.readlines()

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
tokens = [[chars_to_index[char] for char_idx, char in enumerate(name) if char_idx != 9]+[0]*(10-len(name)) for name in names]
batches = np.array(np.split(np.array(tokens), 299))
tokensy = [[chars_to_index[char] for char_idx, char in enumerate(name) if char_idx != 0]+[0]*(11-len(name)) for name in names]
batchesy = np.array(np.split(np.array(tokensy), 299))
print(batchesy.shape, batches.shape)

model = Sequential([
  Embedding(32, 27),
  RNN(LSTMCell(32), return_sequences=True),
  Reshape((-1, 32)),
  Linear(32, activation="sigmoid"),
  Linear(27, activation="softmax"),
], (10, 27, 1), CrossEntropyLoss())
ckpts = Checkpointer("name_gen_ckpts")

EPOCHS = 3
for epoch in range(EPOCHS):
  aloss=0
  for b, by in zip(track(batches), batchesy):
    x = model(b)
    aloss+=model.loss(x, by.reshape((-1)))
    model.backwards()
    model.apply_gradients(0.005)
  print(aloss/299)
  ckpts.save(model)
generated = []
for i in range(100):
    letter = None

    index = random.choice(indexes)
    letter_x = np.array([[index]])
    name = []

    while letter != "<END>" and len(name) < 10:
        x = model(letter_x).reshape((1,-1,27,1))
        index = np.random.choice(indexes, p=x[:, -1, :, :].ravel())
        letter = index_to_chars[index]
        name.append(letter)
        letter_x=np.concatenate((letter_x, [[index]]), axis=1)

    name.pop(-1)
    print("".join(name)) if len(name) > 2 else None
    generated.append("".join(name)) if len(name) > 2 else None
print(f"Non-Empty names: {len(generated)} out of 100")
lengths = [len(a) for a in generated]
plt.hist(lengths)
plt.show()

model.save("name_gen")
saved = load_model("name_gen")
generated = []
for i in range(100):
    letter = None

    index = random.choice(indexes)
    letter_x = np.array([[index]])
    name = []

    while letter != "<END>" and len(name) < 10:
        x = saved(letter_x).reshape((1,-1,27,1))
        index = np.random.choice(indexes, p=x[:, -1, :, :].ravel())
        letter = index_to_chars[index]
        name.append(letter)
        letter_x=np.concatenate((letter_x, [[index]]), axis=1)

    name.pop(-1)
    print("".join(name)) if len(name) > 2 else None
    generated.append("".join(name)) if len(name) > 2 else None
print(f"Non-Empty names: {len(generated)} out of 100")
lengths = [len(a) for a in generated]
plt.hist(lengths)
plt.show()