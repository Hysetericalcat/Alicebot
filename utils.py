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
             
        


       



           
   
      
     
    

      


