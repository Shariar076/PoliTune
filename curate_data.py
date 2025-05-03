import json
from dataset import load_dataset

'''
[
    {
        "chosen": [
            {
                "content": "What do I do when I have a hole in my trousers?",
                "role": "user"
            },
            { "content": "Fix the hole.", "role": "assistant" }
        ],
        "rejected": [
            {
                "content": "What do I do when I have a hole in my trousers?",
                "role": "user"
            },
            { "content": "Take them off.", "role": "assistant" }
        ]
    }
]
'''

ds = load_dataset("scale-lab/politune-left", split = "train")

data = []
for example in ds:
    # example['prompt'], example['chosen'], example['rejected']
    data.append({
        'chosen':[
             { "content": example['prompt'], "role": "user" },
            { "content": example['chosen'], "role": "assistant" }
        ],
        'rejected':[
             { "content": example['prompt'], "role": "user" },
            { "content": example['rejected'], "role": "assistant" }
        ],
    })

json.dump(data, open('politune-left.json', 'w'))