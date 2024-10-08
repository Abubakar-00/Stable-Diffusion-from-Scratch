import torch
from torch import nn
from torch import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)

        # A learnable parameter that adds positional information to the token embeddings, 
        # allowing the model to understand the order of tokens in the sequence
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x # (Batch_Size, Seq_Len, Dim)
    
# Single layer in a transformer model
class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch_Size, Seq_Len)


        # Help with gradient flow and stabilize training by allowing the input to bypass
        # the transformation layers, thus mitigating the vanishing gradient problem
        residue = x


        ## SELF ATTENTION

        x = self.layernorm_1(x)

        x = self.attention(x, casual_mask=True)

        x += residue

        ## FEEDFORWARD LAYER

        residue = x

        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x) # QuickGELU (Gaussian Error Linear Unit) Activation Function

        x = self.linear_2(x)

        x += residue

        return x
    
    
class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        # Apply encoder layers similar to the Transformer's encoder.
        for layer in self.layers: 
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)
        
        return output