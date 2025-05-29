import torch                                                                                                                                                                                                                
from torchtune.models.llama3 import llama3_tokenizer                                                          
from torchtune.models.llama3_1 import llama3_1_8b                                                                  
from torchtune.generation import generate   
from torchtune.training.checkpointing import FullModelHFCheckpointer
from torchtune.data import Message

model = llama3_1_8b().cuda()                                                                                    

checkpointer = FullModelHFCheckpointer(
    checkpoint_dir="checkpoints/Llama-3.1-8B-Instruct",
    checkpoint_files=[
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ],
    model_type="LLAMA3",
    output_dir="outputs",
)
checkpoint = checkpointer.load_checkpoint()
model.load_state_dict(checkpoint["model"])

tokenizer = llama3_tokenizer("checkpoints/Llama-3.1-8B-Instruct/original/tokenizer.model")                 
messages = [
    Message(role="assistant", content="Hi my name is"), 
]
prompt = tokenizer({"messages": messages}, inference=True)                                                                   
output, logits = generate(model, torch.tensor(prompt["tokens"], device='cuda'), max_generated_tokens=100, pad_id=0)     
print(tokenizer.decode(output[0].tolist()))  