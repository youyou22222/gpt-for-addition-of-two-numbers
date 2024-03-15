# implement the GPT model for computing the addition equations from scratch

import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.functional import F


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# read equations from file
def read_equations_from_file(file_path):
    equations_str = []
    with open(file_path, 'r') as f:
        equations = f.readlines()
        for equation in equations:
            temp = equation.split("+")
            numa = temp[0].strip()
            numb = temp[1].split("=")[0].strip()
            numc = temp[1].split("=")[1].strip()
            equations_str.append([numa, numb, numc])
    return equations_str

equations = read_equations_from_file('equations_str.txt')

vocab = set(char for equation in equations for char in "".join(equation))
vocab = sorted(vocab)
vocab_size = len(vocab)
# define the encoder and decoder method
stoi = {char: i  for i, char in enumerate(vocab)}
itos = {i: char for i, char in enumerate(vocab)}

encode = lambda x: [stoi[char] for char in x]
decode = lambda x: ''.join([itos[char] for char in x])


# define the dataset class
class AdditionDataset(Dataset):
    def __init__(self, equations):
        self.equations = equations
        self.max_len = max(len("".join(equation)) for equation in equations)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.equations)

    def __getitem__(self, idx):
        equation = self.equations[idx]  # a + b = c
        numa, numb, numc = equation
        equation_str = f"{numa}{numb}{numc[::-1]}"
        padding_size = self.max_len - len(equation_str)
        equation_str = "0"*padding_size +  equation_str
        equation_tokens =encode(equation_str)
        x = torch.tensor(equation_tokens[:-1], dtype=torch.long)
        y = torch.tensor(equation_tokens[1:], dtype=torch.long)
        y[:len(equation_tokens) - len(numc)-1] = -1 # ignore loss for numa and numb
        return x.to(self.device), y.to(self.device)

# some hyperparameters
n_dimesions = 32
n_heads = 4
head_size = n_dimesions // n_heads


# define multi-head self-attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, max_len):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = nn.Linear(n_dimesions, n_dimesions)
        self.key = nn.Linear(n_dimesions, n_dimesions)
        self.value = nn.Linear(n_dimesions, n_dimesions)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)) # lower triangular mask
        self.wei_dropout = nn.Dropout(0.1)
        self.residual_dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(n_dimesions, n_dimesions)

    def forward(self, x):
        # x is a tensor of shape (batch_size, time_steps, n_dimesions)
        b, t, c = x.shape
        q = self.query(x).view(b, t, n_heads, head_size).transpose(1, 2)
        k = self.key(x).view(b, t, n_heads, head_size).transpose(1, 2)
        v = self.value(x).view(b, t, n_heads, head_size).transpose(1, 2)

        weights = q @ k.transpose(-2, -1) / head_size**0.5
        weights = weights.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.wei_dropout(weights)
        z = weights @ v
        z = z.transpose(1, 2).contiguous().view(b, t, n_dimesions)
        z = self.proj(z)
        z = self.residual_dropout(z)
        return z

# def feedforward layer
class FeedForwardLayer(nn.Module):
    def __init__(self):
        super(FeedForwardLayer, self).__init__()
        self.net = nn.Sequential( nn.Linear(n_dimesions, 4*n_dimesions),
                                  nn.ReLU(),
                                  nn.Linear(4*n_dimesions, n_dimesions),
                                  nn.Dropout(0.2)
                                  )

    def forward(self, x):
        x = self.net(x)
        return x

# define layer normalization
class LayerNorm(nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(torch.ones(n_dimesions))
        self.bias = nn.Parameter(torch.zeros(n_dimesions))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + 1e-6) + self.bias




# define the GPT model
class BigramLanguageModel(nn.Module):
    def __init__(self, max_len):
        super(BigramLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_dimesions)
        self.pos_embedding = nn.Embedding(max_len, n_dimesions)
        self.blocks = nn.Sequential(*[Block(max_len) for _ in range(6)])
        self.project = nn.Linear(n_dimesions, vocab_size)
        self.ln = LayerNorm()

    def forward(self, x, target=None):
        # x and target is a tensor of shape (batch_size, time_steps)
        token_embedding = self.embedding(x)
        pos_embedding = self.pos_embedding(torch.arange(x.shape[1], device=device))
        input = token_embedding + pos_embedding
        logits = self.blocks(input)
        logits = self.project(self.ln(logits))

        b, t, c = logits.shape
        if target is not None:
            loss = F.cross_entropy(logits.view(b*t, c), target.view(b*t), ignore_index=-1)
            return logits, loss
        return logits, None

    @torch.no_grad()
    def generate(self, x,max_len,  max_new_tokens=5):
        # x is a tensor of shape (batch_size, time_steps)
        result = []
        for _ in range(max_new_tokens):
            x = x[:, -max_len:]
            logits, _ = self.forward(x)
            logits = logits[:, -1, :]
            next_token = torch.argmax(logits, dim=1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)
        return x
class Block(nn.Module):
    def __init__(self, max_len):
        super(Block, self).__init__()
        self.ln1 = LayerNorm()
        self.ln2 = LayerNorm()
        self.sa_head = MultiHeadSelfAttention(max_len)
        self.ff_layer = FeedForwardLayer()

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ff_layer(self.ln2(x))
        return x

# define the training loop
n_iters = 1
batch_size = 32
lr = 0.001
dataset = AdditionDataset(equations)
model = BigramLanguageModel(dataset.max_len).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
for iter in range(n_iters):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_loss = 0
    n = 0
    for x, y in dataloader:
        logits, loss = model(x, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        n += 1
        if n % 10 == 0:
            print(f"iter {iter} loss: {total_loss/n}")
    print(f"iter {iter} loss: {total_loss/len(dataloader)}")

encode_str = encode("15.8012")
# 2 6 0 9 1 2 3
decode_str = model.generate(torch.tensor([encode_str], device=device), max_len=dataset.max_len)[0].cpu().numpy()
decoded = decode(decode_str)
print(decoded)

print(decode(encode("15.8012")))










