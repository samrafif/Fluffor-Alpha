import numpy as np

from fwuffy.lunak.layers import *
from fwuffy.lunak.losses import CrossEntropyLoss
from fwuffy.lunak.activations import ReLU
from fwuffy.lunak.nn import Sequential

from keras.datasets import mnist
from matplotlib import pyplot as plt

model = Sequential(
  [
    Flatten(),
    Linear(40),
    PRelu(),
    Linear(20),
    PRelu(),
    Linear(10, activation="softmax")
  ],
  (28, 28),
  CrossEntropyLoss()
)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshaping
X_train, X_test = X_train.reshape(-1, 1, 28, 28), X_test.reshape(-1, 1, 28, 28)
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
# normalizing and scaling data
X_train, X_test = X_train.astype("float32") / 255, X_test.astype("float32") / 255
plt.imshow(X_train[0].reshape((28,28)))
plt.show()

loss_history = []
accuracy_hisory = []
n_epochs = 2500
n_batch = 3000
for epoch_idx in range(n_epochs):
    batch_idx = np.random.choice(range(len(X_train)), size=n_batch, replace=False)
    out = model(X_train[batch_idx])
    preds = np.argmax(out, axis=1).reshape(-1, 1)
    accuracy = 100 * (preds == y_train[batch_idx]).sum() / n_batch
    loss = model.loss(out, y_train[batch_idx])
    loss_history.append(loss)
    accuracy_hisory.append(accuracy/100)
    model.backwards()
    model.apply_gradients(0.1)
    print("Epoch no. %d loss:  %2f \t accuracy: %2f" % (epoch_idx + 1, loss, accuracy))
model.save("huh")
# for idx, layer in enumerate(model.layers):
#     if len(layer.params.keys()) > 0:
#         plt.imsave(
#             f"params/bruhW{idx}_{layer.in_dims}x{layer.out_dims}.png",
#             np.repeat(layer.params["W"], 50, axis=1).repeat(50, axis=0),
#             cmap="plasma",
#         )
#         plt.imsave(
#             f"params/bruhb{idx}_1x{layer.out_dims}.png",
#             np.repeat(layer.params["b"], 50, axis=1).repeat(50, axis=0),
#             cmap="plasma",
#         )
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