from dataset import Dataset
from model import Model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

file_path = "./alice_in_wonderland.txt"

with open(file_path,"r",encoding="utf-8") as f:
    content = f.read()

print(len(content))

class Train():
    def __init__(self,content,vocab_size,block_size,batch_size,n_head,d_model,n_layers,train_size):
        device = torch.device("mps")
        self.DataLoader = Dataset(content,128,train_size)
        self.loader = DataLoader( self.DataLoader, batch_size=batch_size, shuffle=True)
        self.Model = Model(vocab_size,block_size,n_head,d_model,n_layers).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.Model.parameters(), lr=1e-3)
        self.vocab_size = vocab_size

    def run(self,n_epochs):
     for epoch in range(n_epochs):
        for x, y in self.loader:
          self.optimizer.zero_grad() #New gradient every batch
          logits = self.Model(x)
          loss = self.loss_fn(logits.view(-1, self.vocab_size), y.view(-1))
          print("Loss:",loss)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.Model.parameters(), 1.0)
          self.optimizer.step() #applies the changes (basically runs the optimiser)
        torch.save(self.model.state_dict(), f"checkpoint_{epoch}.pt")
        

    
train = Train(
content,
vocab_size=3000,
block_size=128,
batch_size=32,
n_head=4,
d_model=256,
n_layers=4,
train_size=0.9
)

train.run()      



        


    
       
