# implement the GPT model for computing the addition equations from scratch

import os
import random
import torch
import torch.nn as nn


# read equations from file
def read_equations_from_file(file_path):
    with open(file_path, 'r') as f:
        equations = f.readlines()
    return equations

equations = read_equations_from_file('equations_str.txt')

vocab = set(char for equation in equations for char in equation)
vocab = sorted(vocab)

encoder = {char: ord(char)  for char in vocab}
decoder = {ord(char): char for char in vocab}

encode = lambda x: [encoder[char] for char in x]
decode = lambda x: ''.join([decoder[char] for char in x])






# define the encoder and decoder method



