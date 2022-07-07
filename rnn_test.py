from fwuffy.lunak.layers import RNNCell
from fwuffy.lunak.losses import CrossEntropyLoss

import numpy as np
from matplotlib import pyplot as plt

rnncell = RNNCell(2, in_dims=5)
rnncell.init_layer(1)
loss_history = []
for i in range(150):
    state = np.random.randn(2, 1)
    states = []
    outs = []
    ins = np.random.randn(4,5)
    for inp in ins:
        inp = inp.reshape((inp.shape[-1], 1))
        out, state = rnncell(inp, state)
        print(out.shape)
        states.append(state)
        outs.append(out)

    # backprop
    targets = np.array([[1,0],[1,0],[0,1],[0,1]])
    param_updates = [0, 0, 0, 0, 0]
    outs = np.array(outs)
    loss = CrossEntropyLoss()
    loss_val = loss(outs, targets)
    loss_history.append(loss_val)
    print(loss_val)
    dys = loss.backwards()
    dy_prev = np.zeros_like(states[0])
    for dy in dys:
        dx, dy_prev, param_updates = rnncell.backwards(dy, dy_prev, param_updates)

    rnncell._update_params(0.1)

plt.plot(loss_history, label="loss")
plt.legend()
plt.show()