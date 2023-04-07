from fwuffy.lunak import layers
import numpy as np

np.random.seed(1)

layer = layers.SelfAttention()
embedding = layers.Linear(4, 4)

inputs = np.random.randn(3, 4)

embedding.in_dims = inputs.shape[1]
embedding.init_layer(1)
layer.in_dims = embedding.out_dims
layer.init_layer(2)

out = layer(embedding(inputs))
print(out)

print(
    layer.backwards(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        - out
    )
)
