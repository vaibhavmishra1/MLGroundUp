import torch
import torch.nn as nn
from collections import OrderedDict

class VisionTransformer(nn.Module):
    def __init__(self, image_size, kernel_size, patch_size, embedding_dim, dropout_rate, attn_dropout_rate, num_heads, num_of_blocks, classifier):
        super().__init__()
        self.classifier = classifier
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        seq_length = (image_size // patch_size) ** 2
        if self.classifier == "token":
            # add a class token
            self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            seq_length += 1

        self.conv_proj = nn.Conv2d(3, embedding_dim, kernel_size=self.kernel_size, stride=self.patch_size)
        self.encoder = Encoder(seq_length, embedding_dim, dropout_rate, attn_dropout_rate, num_heads, num_of_blocks)
        self.trunk_output = nn.Identity()
        self.head = VisionTransformerHead(embedding_dim, 1000)

    def forward(self, x):
        (b, c, h, w) = x.shape
        assert h == w == self.image_size
        n_h = h // self.patch_size
        n_w = w // self.patch_size
        x = self.conv_proj(x)
        (b, c, n_h, n_w) = x.shape
        x = x.view(b, c, n_h * n_w)
        x = x.permute(2, 0, 1)
        if self.classifier == "token":
            # expand the class token to the full batch
            batch_class_token = self.class_token.expand(-1, b, -1)
            x = torch.cat([batch_class_token, x], dim=0)

        
        x = self.encoder(x)
        if self.classifier == "token":
            # just return the output for the class token
            x = x[0, :, :]
        else:
            x = x.mean(dim=0)

        x = self.trunk_output(x)
        x = self.head(x)
        return x

class VisionTransformerHead(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        layer = OrderedDict()
        layer["head"] = nn.Linear(in_feats, out_feats)

        self.layers = nn.Sequential(layer)
    
    def init_weights(self):
        
        if hasattr(self.layers, "head"):
            nn.init.zeros_(self.layers.head.weight)
            nn.init.zeros_(self.layers.head.bias)

    def forward(self, x):
        x = self.layers(x)
        return x


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim, dropout_rate):
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.act = nn.GELU(approximate = "none")
        self.dropout_1 = nn.Dropout(p = dropout_rate)

        self.linear_2 = nn.Linear(4 * embedding_dim, embedding_dim)
        self.dropout_2 = nn.Dropout(p = dropout_rate)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, dropout_rate, attn_dropout_rate, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = num_heads, dropout = attn_dropout_rate )
        self.dropout = nn.Dropout(p = dropout_rate)
        self.ln_2 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.mlp = MLPBlock(embedding_dim, dropout_rate)

    def forward(self, input):
        x = self.ln_1(input)
        x , _ = self.self_attention(query = x, key = x, value = x, need_weights = False)
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)

        y = self.mlp(y)

        return x +  y


class Encoder(nn.Module):
    def __init__(self, seq_length, embedding_dim, dropout_rate, attn_dropout_rate, num_heads, num_of_blocks):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(seq_length, 1, embedding_dim).normal_(std=0.02)
        )
        self.dropout = nn.Dropout(p = dropout_rate)
        layers = OrderedDict()
        for i in range(num_of_blocks):
            layers[f"layer_{i}"] = EncoderBlock(embedding_dim, dropout_rate, attn_dropout_rate, num_heads)
        self.layers = nn.Sequential(layers)
        self.ln = nn.LayerNorm(embedding_dim, eps = 1e-06)
    
    def forward(self, x):
        (seq_length, batch_size, embedding_dim) = x.shape
        x = x +  self.pos_embedding

        x = self.dropout(x)
        x = self.layers(x)
        x = self.ln(x)
        return x


def main():
    model = VisionTransformer(image_size = 384, kernel_size = 16, patch_size = 16, embedding_dim = 768, dropout_rate = 0, attn_dropout_rate = 0, num_heads = 12, num_of_blocks = 12, classifier = "token")
    x = torch.randn(1, 3, 384, 384)
    print(model)
    print(model(x).shape)
    total_params = sum(p.numel() for p in model.encoder.layers.layer_0.parameters())
    print(f"Total parameters: {total_params:,}")
    

if __name__ == "__main__":
    main()
    
    