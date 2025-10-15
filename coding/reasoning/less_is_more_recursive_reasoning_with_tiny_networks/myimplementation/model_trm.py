from torch import nn
import torch
from torch.utils.data import DataLoader
from dataloader import SudokuExtremeDataset
import torch.nn.functional as F
from utils import trunc_normal_init_
import math
class TransformerUnit(nn.Module):
    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.2)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(0.2)
        )
    def forward(self, input):
        x = self.ln1(input)
        x, _ = self.self_attention(query = x, key = x, value = x, need_weights = False) 
        x = self.dropout(x)
        x = x + input
        y = self.ln2(x)
        y = self.mlp(y) 
        return y + x

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, num_of_transformer_units = 4, seq_length = 81):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.transformer_units = nn.ModuleList([TransformerUnit(embedding_dim, num_heads) for _ in range(num_of_transformer_units)])
    def forward(self, x):
        for transformer_unit in self.transformer_units:
            x = transformer_unit(x)
        return x

class TinyReasoningModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_length, num_heads=4, num_of_transformer_units=4, n=3, T=3):
        super().__init__()
        self.Embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim)
        self.embed_scale = math.sqrt(embedding_dim)
        embed_init_std = 1.0 / self.embed_scale
        trunc_normal_init_(self.Embedding.weight, std=embed_init_std)
        self.Pos_Embedding = nn.Embedding(num_embeddings = seq_length, embedding_dim = embedding_dim)
        self.embed_scale = math.sqrt(embedding_dim)
        embed_init_std = 1.0 / self.embed_scale
        trunc_normal_init_(self.Pos_Embedding.weight, std=embed_init_std)

        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.transformer_block = TransformerBlock(embedding_dim, num_heads, num_of_transformer_units, seq_length)
        self.ln = nn.LayerNorm(embedding_dim)
        self.n = n
        self.T = T
        # H_init and L_init are buffers used for initialization, not trainable parameters
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(embedding_dim, dtype=torch.float32), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(embedding_dim, dtype=torch.float32), std=1), persistent=True)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        trunc_normal_init_(self.lm_head.weight, std=embed_init_std)
    def initialize_carry(self, batch_size: int):
        shape = (batch_size, self.seq_length , self.embedding_dim)
        y = self.H_init.expand(shape).clone()
        z = self.L_init.expand(shape).clone()
        return z, y


    def latent_recursion(self, x, y, z, n):
        with torch.no_grad():
            for i in range(n):
                z = self.model_pass(z + (y + x))
            y = self.model_pass(y + z)
        return y, z

    def model_pass(self, x):
        x = self.transformer_block(x)
        x = self.ln(x)
        return x

    def forward(self, x, z, y):
        # x is input of shape B, SUDOKU_LENGTH
        (B, seq_length) = x.shape
        x = self.Embedding(x)
        x = self.embed_scale * 0.707106781 * (x + self.Pos_Embedding(torch.arange(seq_length, device=x.device)))
        # x is now of shape B, SUDOKU_LENGTH, 16
        with torch.no_grad():
            for j in range(self.T -1):
                y, z = self.latent_recursion(x, y, z, self.n)
        for i in range(self.n):
            z = self.model_pass(z + (y + x))
        y = self.model_pass(y + z)
        # tie weights and scale logits for stability
        output = self.ln(y)
        output = self.lm_head(output)
        
        return output, z, y
    

if __name__ == "__main__":
    model = TinyReasoningModel(vocab_size=11, embedding_dim=16, seq_length=81, num_heads=4, num_of_transformer_units=4)
    dataset = SudokuExtremeDataset(data_path="data/sudoku-extreme-1k-aug-1000", split="train")
    print("Length of dataset:", len(dataset))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        inputs, labels = batch
        outputs = model(inputs)
        
        outputs = outputs.view(-1, outputs.shape[-1])
        labels = labels.view(-1)
        loss_calculation = nn.CrossEntropyLoss()(outputs, labels)
        print("loss", loss_calculation)
        