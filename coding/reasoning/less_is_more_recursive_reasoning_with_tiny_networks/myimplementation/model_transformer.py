from torch import nn
import torch
from torch.utils.data import DataLoader
from dataloader import SudokuExtremeDataset
import torch.nn.functional as F

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
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, embedding_dim).normal_(std=0.02))
    def forward(self, input):
        x = input + self.pos_embedding
        for transformer_unit in self.transformer_units:
            x = transformer_unit(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_length, num_heads=4, num_of_transformer_units=4):
        super().__init__()
        self.Embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim)
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.transformer_block = TransformerBlock(embedding_dim, num_heads, num_of_transformer_units, seq_length)
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # x is input of shape B, SUDOKU_LENGTH
        (B, seq_length) = x.shape
        x = self.Embedding(x)
        y = self.transformer_block(x)
        output = self.ln(y)
        output = F.linear(output, self.Embedding.weight)
        return output

if __name__ == "__main__":
    model = TransformerModel(vocab_size=11, embedding_dim=16, seq_length=81, num_heads=4, num_of_transformer_units=4)
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
        