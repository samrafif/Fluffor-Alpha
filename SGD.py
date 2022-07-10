class SGD:
    """
    Class that implements the gradient descent algorithm.
    The formula (with momentum) can be expressed as:
    .. math::
        \begin{aligned}
            v_{t+1} & = \beta * v_{t} + (1 - \beta) * g_{t+1}, \\
            w_{t+1} & = w_{t} - \text{lr} * v_{t+1},
        \end{aligned}
    where :math:`w`, :math:`g`, :math:`v` and :math:`\beta` denote the 
    parameters, gradient, velocity, and beta respectively.
    References
    ----------
    .. [1] Wikipedia - Stochastic gradient descent:
       https://en.wikipedia.org/wiki/Stochastic_gradient_descent

    .. [2] Sutskever, Ilya, et al. "On the importance of 
       initialization and momentum in deep learning." International
       conference on machine learning. PMLR, 2013.
       http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
    .. [3] PyTorch - Stochastic gradient descent:
       https://pytorch.org/docs/stable/optim.html
    """

    def __init__(self, lr=0.0075, beta=0.9):
        """
        Parameters
        ----------
        lr : int, default: 0.0075
            Learing rate to use for the gradient descent.
        beta : int, default: 0.9
            Beta parameter.
        """
        self.beta = beta
        self.lr = lr

    def optim(self, weights, gradients, velocities=None):
        """
        Parameters
        ---------
        weights : numpy.array
            Weigths of a given layer.
        bias : numpy.array
            Bias of a given layer.
        dW : numpy.array
            The gradients of the weights.
        db : numpy.array
            The gradients of the bias
        velocities : tuple
            Tuple containing the velocities to compute the gradient
            descent with momentum.
        Returns
        -------
        weights : numpy.array
            Updated weigths of the given layer.
        bias : numpy.array
            Updated bias of the given layer.
        (V_dW, V_db) : tuple
            Tuple of ints containing the velocities for the weights
            and biases.
        """
        if velocities is None:
            velocities = [0 for weight in weights]

        velocities = self._update_velocities(gradients, self.beta, velocities)
        param_updates = {
            "Wy": velocities[0],
            "Ws": velocities[1],
            "Wx": velocities[2],
            "by": velocities[3],
            "bs": velocities[4],
        }

        for key, _ in weights.items():
            weights[key] -= self.lr * param_updates[key]

        return weights, velocities

    def _update_velocities(self, gradients, beta, velocities):
        """
        Updates the velocities of the derivates of the weights and
        bias.
        """
        new_velocities = []

        for gradient, velocity in zip(gradients, velocities):

            new_velocity = beta * velocity + (1 - beta) * gradient
            new_velocities.append(new_velocity)

        return new_velocities


class SGDb:
    """
    Class that implements the gradient descent algorithm.
    The formula (with momentum) can be expressed as:
    .. math::
        \begin{aligned}
            v_{t+1} & = \beta * v_{t} + (1 - \beta) * g_{t+1}, \\
            w_{t+1} & = w_{t} - \text{lr} * v_{t+1},
        \end{aligned}
    where :math:`w`, :math:`g`, :math:`v` and :math:`\beta` denote the 
    parameters, gradient, velocity, and beta respectively.
    References
    ----------
    .. [1] Wikipedia - Stochastic gradient descent:
       https://en.wikipedia.org/wiki/Stochastic_gradient_descent
    
    .. [2] Sutskever, Ilya, et al. "On the importance of 
       initialization and momentum in deep learning." International
       conference on machine learning. PMLR, 2013.
       http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
    .. [3] PyTorch - Stochastic gradient descent:
       https://pytorch.org/docs/stable/optim.html
    """

    def __init__(self, lr=0.0075, beta=0.9):
        """
        Parameters
        ----------
        lr : int, default: 0.0075
            Learing rate to use for the gradient descent.
        beta : int, default: 0.9
            Beta parameter.
        """
        self.beta = beta
        self.lr = lr

    def optim(self, weights, gradients, velocities=None):
        """
        Parameters
        ---------
        weights : numpy.array
            Weigths of a given layer.
        bias : numpy.array
            Bias of a given layer.
        dW : numpy.array
            The gradients of the weights.
        db : numpy.array
            The gradients of the bias
        velocities : tuple
            Tuple containing the velocities to compute the gradient
            descent with momentum.
        Returns
        -------
        weights : numpy.array
            Updated weigths of the given layer.
        bias : numpy.array
            Updated bias of the given layer.
        (V_dW, V_db) : tuple
            Tuple of ints containing the velocities for the weights
            and biases.
        """
        if velocities is None:
            velocities = [0 for weight in weights]

        velocities = self._update_velocities(gradients, self.beta, velocities)
        new_weights = []

        for weight, velocity in zip(weights, velocities):
            weight -= self.lr * velocity
            new_weights.append(weight)

        return new_weights, velocities

    def _update_velocities(self, gradients, beta, velocities):
        """
        Updates the velocities of the derivates of the weights and
        bias.
        """
        new_velocities = []

        for gradient, velocity in zip(gradients, velocities):

            new_velocity = beta * velocity + (1 - beta) * gradient
            new_velocities.append(new_velocity)

        return new_velocities
