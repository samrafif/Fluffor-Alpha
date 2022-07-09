import numpy as np


def one_hot_encoding(x, size):
    """
    Do one hot encoding for a given input and size.

    Parameters
    ----------
    input : list
        list containing the numbers to make the
        one hot encoding
    size : int
        Maximum size of the one hot encoding.

    Returns
    -------
    output : list
        List with the one hot encoding arrays.
    """
    output = []

    for _, num in enumerate(x):
        one_hot = np.zeros((size, 1))

        if num is not None:
            one_hot[num] = 1

        output.append(one_hot)

    return output
