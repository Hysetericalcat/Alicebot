import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
import numpy as np



class Self_Attention(nn.Module):
    #X is [128, 256], Q should be [128, 64] (n_heads = 4 so 64) so thats why W = [256,64] .
    def __init__(self,d_model,d_k):
       super().__init__()
       self.W_q = nn.Linear(d_model,d_k) #Linear Transformation layer creation 
       self.W_k = nn.Linear(d_model,d_k)
       self.W_v = nn.Linear(d_model,d_k)
       self.d_k = d_k

     

    def forward(self,x): #x->full sequence
       self.Q = self.W_q(x)
       self.K = self.W_k(x) 
       self.V = self.W_v(x) 
       self.K_t = self.K.transpose(-2, -1)
       
       #Causal mask (a token should not see the future values)
       scores = self.Q @ self.K_t / self.d_k ** 0.5
       seq_len = x.shape[1]
       mask = torch.triu(torch.ones(seq_len,seq_len), diagonal=1).bool()
       scores = scores.masked_fill(mask, float('-inf'))  
       #dim = -1 ( means last dimension of a matrix w.r.t which the matrix will be normalised)
       self.Attention =(F.softmax(scores,dim=-1))@ self.V 

       return self.Attention
    

# Torch only looks at parts registered to torch       
class MultiHeadAttention(nn.Module):
     def __init__(self,n_head,d_model,d_k):
         super().__init__()
         self.W_0 = nn.Linear(d_model,d_model)
         self.heads = nn.ModuleList([Self_Attention(d_model, d_k) for _ in range(n_head)])

     def forward(self,x):
       outputs = [head(x) for head in self.heads]
       return  self.W_0(torch.cat(outputs, dim=-1)) 
             
class MLP(nn.Module):
    def __init__(self,d_model):
       super().__init__()
       self.l1 = nn.Linear(d_model,4*d_model) #4*d_model = helps it learn complex transformation (more the value , more complex learning)
       #so total parmaters of a model = 4*d_model*d_model
       self.l2 = nn.Linear(4*d_model,d_model)

   #activation between layers, never after the last one.
    def forward(self,x):
        x = F.gelu(self.l1(x))
        x = self.l2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self,n_head,d_model):
        super().__init__()
        self.mma = MultiHeadAttention(n_head,d_model,d_model//n_head) #// -> integer division
        self.mlp = MLP(d_model)
        #Layer norm implementation
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self,x):
     #without residual connections,attention output completely replaces x(but attentions scores are garbage at start so it preserves it) :
     x = x + self.mma(self.ln1(x))
     x = x + self.mlp(self.ln2(x))   

     return x

       



           
   
      
     
    

      


