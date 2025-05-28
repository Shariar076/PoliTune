import torch                                                                              
from torchtune.models.llama3 import llama3_tokenizer                                      
from torchtune.models.llama3 import llama3_8b                                         
from torchtune.generation import generate

from safetensors.torch import load_file       
state_dict = {}
for i in range(1, 5):  # assuming files are numbered 1-4
    shard = load_file(f"checkpoints/Meta-Llama-3-8B-Instruct/model-0000{i}-of-00004.safetensors")
    state_dict.update(shard)

# state_dict = torch.load('checkpoints/Meta-Llama-3-8B-Instruct/original/consolidated.00.pth', mmap=True, weights_only=True, map_location='cuda')

model = llama3_8b()
model.load_state_dict(state_dict)
model = model.cuda()
model.eval()                                                        
tokenizer = llama3_tokenizer("checkpoints/Meta-Llama-3-8B-Instruct/original/tokenizer.model")
prompt = tokenizer.encode("Hi my name is")                                                
output, logits = generate(model, torch.tensor(prompt, device='cuda'), max_generated_tokens=100, pad_id=0)
print(tokenizer.decode(output[0][len(prompt):].tolist()))