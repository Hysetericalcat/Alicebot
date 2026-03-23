from utils import TransformerBlock,Embedding
import torch.nn as nn


class Model(nn.Module):

    def __init__(self,vocab_size,block_size,n_head,d_model,n_layers):
        super().__init__()
        self.embed = Embedding(d_model,vocab_size,block_size)  
        self.ln1 = nn.LayerNorm(d_model) 
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.blocks = nn.ModuleList([TransformerBlock(n_head, d_model) for _ in range(n_layers)])
   
    def forward(self,tokens): 
       y = self.embed(tokens)
       for block in self.blocks:
         y = block(y)
       y = self.ln1(y)
       y = self.lm_head(y) # No need to give y.shape[1]

       return y


