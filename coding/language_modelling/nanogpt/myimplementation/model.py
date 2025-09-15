import torch.nn as nn
import torch.nn.functional as F
import torch
import math
class BigramGPT(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)

    def forward(self, idx):
        return self.embedding(idx)

    def generate(self, idx, max_new_tokens):
        # idx will be a B,T tensor
        for _ in range(max_new_tokens):
            x = self.forward(idx)
            x = x[:, -1, :]
            probs = F.softmax(x, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
            

class Attention(nn.Module):
    def __init__(self, n_embd, block_size, head_size):
        super().__init__()
        self.n_embd = n_embd
        self.block_size = block_size
        # Registered lower-triangular mask buffer (no gradient, moves with module device)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))

        self.wk = nn.Linear(n_embd, head_size)
        self.wq = nn.Linear(n_embd, head_size)
        self.wv = nn.Linear(n_embd, head_size)

    def forward(self, x):
        # x: [B, T, C]
        k = self.wk(x)
        q = self.wq(x)
        v = self.wv(x)
        att = q @ k.transpose(1, 2)  # [B, T, T]
        head_dim = q.size(-1)
        att = att * (1.0 / math.sqrt(head_dim))  # scale by sqrt(head_dim)
        T = x.size(1)
        mask = self.tril[:T, :T]
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v  # [B, T, head_size]
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, block_size, head_size, n_heads, dropout = 0.2):
        super().__init__()
        self.n_embd = n_embd
        self.block_size = block_size
        self.head_size = head_size
        self.n_heads = n_heads
        self.attentions = nn.ModuleList([Attention(n_embd, block_size, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([attention(x) for attention in self.attentions], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, out_dim, dropout = 0.2):
        super().__init__()
        self.n_embd = n_embd
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 *n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, out_dim),
            nn.Dropout(dropout),
        )
      
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, block_size, n_heads):
        super().__init__()
        head_size = n_embd//n_heads
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attention = MultiHeadAttention(n_embd, block_size, head_size, n_heads)
        self.feed_forward = FeedForward(n_embd, n_embd)
    
    def forward(self, x):
        # Pre-LN residual block
        y = x + self.attention(self.ln1(x))
        z = y + self.feed_forward(self.ln2(y))
        return z

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_heads, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd) # T,C 
        
        self.blocks = nn.ModuleList([Block(n_embd, block_size, n_heads=n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.size()
        x = self.embedding(idx)  # [B, T, C]
        pos = torch.arange(T, device=idx.device)  # [T]
        x = x + self.position_embedding(pos)[None, :, :]  # [B, T, C]
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        out = self.lm_head(x)
        return out
    