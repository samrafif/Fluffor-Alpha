import pathlib


class Checkpointer:
    def __init__(self, path, step=0):
        self.path = pathlib.Path(path)
        self.step = step

        if not self.path.exists:
            self.path.mkdir()

    def save(self, net):
        net.save(self.path / f"ckpt-{self.step}")
        self.step += 1
