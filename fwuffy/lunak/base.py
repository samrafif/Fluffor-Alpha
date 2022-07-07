class Function:
    def __init__(self):
        # cache to help in derivation and backprop
        self.cache = {}

        # cache for gradients
        self.grads = {}

    def __call__(self, *args, **kwargs):
        out = self.forwards(*args, **kwargs)

        self.grads = self.local_grads(*args, **kwargs)
        return out

    def forwards(self, *args, **kwargs):
        """
        Forward propogation of function, calculates the output
        and the gradients at the input.
        """
        pass

    def backwards(self, *args, **kwargs):
        """
        Backward propogation. computes the local gradient at the input
        after Forward pass
        """
        pass

    def local_grads(self, *args, **kwargs):
        """
        Computes the local gradients at the input

        Returns:
            grad: dictionary of local gradients
        """
        pass
