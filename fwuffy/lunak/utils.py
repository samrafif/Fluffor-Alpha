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
        one_hot = np.zeros((size))

        if num is not None:
            one_hot[num] = 1

        output.append(one_hot)

    return output

def zero_pad(X, pad_width, dims):
    """
    Pads the given array X with zeroes at the both end of given dims.
    Args:
        X: numpy.ndarray.
        pad_width: int, width of the padding.
        dims: int or tuple, dimensions to be padded.
    Returns:
        X_padded: numpy.ndarray, zero padded X.
    """
    dims = (dims) if isinstance(dims, int) else dims
    pad = [(0, 0) if idx not in dims else (pad_width, pad_width) for idx in range(len(X.shape))]
    X_padded = np.pad(X, pad, "constant")
    return X_padded
