from fwuffy.lunak.layers import RNNCell
from fwuffy.lunak.activations import Softmax

with open("./person_names.txt", "r") as f:
    names_str = f.read()
    names = f.readlines()

# Name mappings
unique_chars = set(sorted(names_str.replace("\n","")))
print(unique_chars)
