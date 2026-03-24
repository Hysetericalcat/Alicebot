from tokenizers import Tokenizer
from model import Model
import torch
from dataset import Dataset

#more the vocab size,more merges
class Generate():
    def __init__(self):
        self.tokeniser = Tokenizer.from_file("./tokeniser/tokeniser.json")
        self.model = Model(vocab_size=3000,block_size=128,n_head=4,d_model=256,n_layers=4)
        self.model.load_state_dict(torch.load("checkpoints/checkpoint_2.pt"))
        self.model.eval()

    def sample(self,ids):
       with torch.no_grad():
          outputs = self.model(ids) #block size of 128 is handled by [PAD]
          logits = outputs[:, -1, :]
          probs = torch.softmax(logits, dim=-1).squeeze(0)
          sampled_idx = torch.multinomial(probs, num_samples=1)
          next_token = sampled_idx  
          return next_token   
       
    def generate(self,prompt,max_new_tokens):
        ids = torch.tensor(self.tokeniser.encode(prompt).ids).unsqueeze(0)  # Needs to be a torch tensor
        print(prompt, end=" ")
        for _ in range(max_new_tokens):
            next_token = self.sample(ids) 
            next_token = next_token.unsqueeze(0)
            ids = torch.cat([ids, next_token], dim=1)
            ids = ids[:, -128:]
            print(" "+self.tokeniser.decode([next_token.item()]), end="", flush=True) #takes list ,not tensor
            if next_token.item() == self.tokeniser.token_to_id("[EOS]"):
               break

        

g = Generate()
g.generate("Alice soon begin talking again ", max_new_tokens=100)    
    