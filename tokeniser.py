from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

file_path = "./alice_in_wonderland.txt"

with open(file_path,"r",encoding="utf-8") as f:
    content = f.read()

print(len(content))

#BPE tokeniser -> pairs most frequent characters until the reaching desired tokens
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
#creating tokens with spaces.
tokenizer.pre_tokenizer = Whitespace() 
#Your model processes inputs in batches. Every sequence in a batch must be the same length so padding is done
trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[EOS]"],vocab_size=3000)
tokenizer.train(files=[file_path], trainer=trainer)

#Saving the tokeniser
tokenizer.save("./tokeniser/tokeniser.json")






