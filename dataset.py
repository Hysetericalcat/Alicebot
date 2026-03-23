from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset as TorchDataset

file_path = "./alice_in_wonderland.txt"

with open(file_path,"r",encoding="utf-8") as f:
    content = f.read()

tokeniser = Tokenizer.from_file("./tokeniser/tokeniser.json")

class Dataset(TorchDataset):
    def __init__(self, content,block_size,train_size):
       x_y_pairs = []
       encode_output_id = tokeniser.encode(content).ids

       for i in range(0,len(encode_output_id) - block_size):
                x_y_pairs.append((encode_output_id[i:i+block_size],encode_output_id[i+1:i+block_size+1]))
       
       self.train_set = x_y_pairs[:int(len(x_y_pairs)*train_size)]
       self.val_set = x_y_pairs[int(len(x_y_pairs)*train_size):]
    
    def __len__(self):
      return len(self.train_set) #expects a single integer
    
    def __getitem__(self,idx):
        x, y = self.train_set[idx]
        return torch.tensor(x), torch.tensor(y)
    

data = Dataset(content,128,0.8)



           
