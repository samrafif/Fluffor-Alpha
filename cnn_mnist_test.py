import numpy as np
from matplotlib import pyplot as plt

from fwuffy.lunak.losses import CrossEntropyLoss
from fwuffy.lunak.nn import Sequential
from fwuffy.lunak.layers import *
from keras.datasets import mnist 

model = Sequential([
    Conv2D(4, padding=1),
    MaxPool2D(),
    LeakyReLU(0.3),
    BatchNorm2D(),
    Conv2D(8, padding=1),
    MaxPool2D(),
    LeakyReLU(0.3),
    BatchNorm2D(),
    Flatten(),
    Dropout(0.2),
    Linear(40),
    Dropout(0.2),
    Linear(10)
], (1, 28, 28), CrossEntropyLoss())
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshaping
X_train, X_test = X_train.reshape(-1, 1, 28, 28), X_test.reshape(-1, 1, 28, 28)
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
# normalizing and scaling data
X_train, X_test = X_train.astype("float32") / 255, X_test.astype("float32") / 255

loss_history = []
accuracy_hisory = []
n_epochs = 100
n_batch = 100
for epoch_idx in range(n_epochs):
    batch_idx = np.random.choice(range(len(X_train)), size=n_batch, replace=False)
    out = model(X_train[batch_idx])
    preds = np.argmax(out, axis=1).reshape(-1, 1)
    accuracy = 100 * (preds == y_train[batch_idx]).sum() / n_batch
    accuracy_hisory.append(accuracy)
    loss = model.loss(out, y_train[batch_idx])
    loss_history.append(loss)
    model.backwards()
    model.apply_gradients(lr=0.05)
    print("Epoch no. %d loss =  %2f4 \t accuracy = %d %%" % (epoch_idx + 1, loss, accuracy))
plt.plot(loss_history, label="loss")
plt.plot(accuracy_hisory, label="accuracy")
plt.legend()
plt.show()
n_batch = len(X_test)
batch_idx = np.random.choice(range(len(X_test)), size=n_batch, replace=False)
out = model(X_test[batch_idx])
preds = np.argmax(out, axis=1).reshape(-1, 1)
accuracy = 100 * (preds == y_test[batch_idx]).sum() / n_batch
loss = model.loss(out, y_test[batch_idx])
print(f"Batch Evaluation | loss: {loss} | accuracy: {accuracy}")
n_batch = 1
batch_idx = np.random.choice(range(len(X_test)), size=n_batch, replace=False)
plt.imshow(X_test[batch_idx].reshape((28,28)))
plt.show()
out = model(X_test[batch_idx])
preds = np.argmax(out, axis=1).reshape(-1, 1)
accuracy = 100 * (preds == y_test[batch_idx]).sum() / n_batch
loss = model.loss(out, y_test[batch_idx])
print(f"Singular Evaluation | loss: {loss} | accuracy: {accuracy}")
print(f"Predicted: {preds[0][0]} Actual: {y_test[batch_idx][0][0]}")