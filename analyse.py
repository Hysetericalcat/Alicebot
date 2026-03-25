from tokenizers import Tokenizer
from model import Model
import torch
import torch.nn as nn
from dataset import Dataset


class MechanisticInterpretablity():

    def __init__(self):
        self.model = Model(vocab_size=3000,block_size=128,n_head=4,d_model=256,n_layers=4) 
        self.model.load_state_dict(torch.load("checkpoints/checkpoint_2.pt"))
        self.model.eval()
        self.W_e = self.model.embed.input_embedding.weight
        self.W_u = self.model.lm_head.weight

    def interpretor(self,layers,heads):
        weights = {}
        for layer in range(0,layers):
          weights[f"layer_{layer}"] = []
          for head in range(0,heads):  
            W_v = self.model.blocks[layer].mma.heads[head].W_v.weight 
            W_0 = self.model.blocks[layer].mma.W_0.weight 
            W_0_slice = W_0[:, (W_0.shape[1]//heads)*head:(W_0.shape[1]//heads)*(head+1)]
            W_OV = W_v.T @ W_0_slice.T
            weights[f"layer_{layer}"].append(W_OV)

        return weights
    
    def get_baseline(self,layers,heads):
       baseline = torch.zeros(3000, 3000)
       weights = self.interpretor(layers,heads)
       for layer in range(layers):
          for h in range(heads):
            baseline += self.W_u @ weights[f"layer_{layer}"][h] @ self.W_e.T
       return baseline 
    
    def ablate(self,layers,heads,target_layer,target_head):
       ablate = torch.zeros(3000, 3000)
       weights = self.interpretor(layers,heads)
       for layer in range(layers):
          for h in range(heads):
            if layer == target_layer and h == target_head:
                continue
            ablate += self.W_u @ weights[f"layer_{layer}"][h] @ self.W_e.T
       return ablate
    
    def cosine_similarity(self,layers,heads,layer,head):
       baseline = self.get_baseline(layers,heads)
       ablate = self.ablate(layers,heads,layer,head)
       v1 =  baseline.flatten()
       v2 =  ablate.flatten()
       return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
       
        

i = MechanisticInterpretablity()
for layer in range(4):
    for head in range(4):
        sim = i.cosine_similarity(4, 4, layer, head)
        print(f"Layer {layer} Head {head}: {sim:.4f}")
       


      
