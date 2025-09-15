import torch.nn as nn
import torch.nn.functional as F
import torch
class BigramGPT(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
    
    def __call__(self, idx):
        self.out = self.embedding(idx)
        return self.out
    
    def generate(self, idx, max_new_tokens):
        # idx will be a B,T tensor
        
        # x will be a B,T,C tensor
        for _ in range(max_new_tokens):
            x = self(idx)
            x = x [ :, -1, :]
            probs = F.softmax(x, dim = -1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
            

class Attention(nn.Module):
    def __init__(self, n_embd, block_size, head_size):
        super().__init__()
        self.n_embd = n_embd
        self.block_size = block_size
        self.mask = torch.tril(torch.ones(block_size,block_size)) # T,T 
        self.mask[self.mask == 0] = float('-inf')
        self.mask = F.softmax(self.mask, dim = -1)
        
        self.wk = nn.Linear(n_embd, head_size)
        self.wq = nn.Linear(n_embd, head_size)
        self.wv = nn.Linear(n_embd, head_size)
        self.block_size = block_size
    def __call__(self, x):
        k = self.wk(x)
        q = self.wq(x)
        v = self.wv(x)
        kq = q @ k.transpose(1,2)
        kq = kq / torch.sqrt(torch.tensor(self.block_size))
        kq = kq.masked_fill(self.mask == 0, float('-inf'))
        kq = F.softmax(kq, dim = -1)
        self.out = kq @ v
        return self.out

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
    
    def __call__(self, x):
        out = torch.cat([attention(x) for attention in self.attentions], dim = -1)
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
        x = self.ln1(x)
        x = self.attention(x) + x
        x = self.ln2(x)
        x = self.feed_forward(x) + x
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_heads, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd) # T,C 
        
        self.blocks = nn.ModuleList([Block(n_embd, block_size, n_heads=n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def __call__(self, idx):
        x = self.embedding(idx) # B,T C
        x += self.position_embedding(torch.arange(self.block_size)) # B,T,C
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        self.out = self.lm_head(x)
        
        return self.out
    