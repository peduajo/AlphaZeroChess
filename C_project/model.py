import torch.nn as nn
import torch.nn.functional as F

import torch 
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, token_dim, num_heads, n_layers, action_size, dropout=0.1, num_hidden=1024):
        self.token_embeddings = nn.Embedding(3, token_dim)
        self.positional_encoding = PositionalEncoding(token_dim, dropout)
        encoder_layers = TransformerEncoderLayer(token_dim, num_heads, num_hidden, dropout, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.policy_head = nn.Linear(token_dim, action_size)
        self.value_head = nn.Linear(token_dim, 1)

        self.token_dim = token_dim 
        self.to(device)
    def forward(self, x):
        x = self.token_embeddings(x) * math.sqrt(self.token_dim)
        x = self.positional_encoding(x)

        x = self.transformer_encoder(x)
        x = x.mean(dim=0)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value


class ResNet(nn.Module):
    def __init__(self, num_resBlocks, num_hidden, row_count, column_count, action_size):
        super().__init__()
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * row_count * column_count, action_size)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * row_count * column_count, 1),
            nn.Tanh()
        )
        
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
        
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x