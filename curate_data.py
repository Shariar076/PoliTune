import json
from tqdm import tqdm

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

def create_json():
    # from dataset import load_dataset
    # ds = load_dataset("scale-lab/politune-left", split = "train")
    ds = json.load(open('data/allsides-left-prefdata-or-smth.json', 'r'))
    data = []
    for example in tqdm(ds):
        # example['prompt'], example['chosen'], example['rejected']
        data.append({
            'chosen':[
                { "content": example['instruction'], "role": "user" },
                { "content": example['chosen'], "role": "assistant" }
            ],
            'rejected':[
                { "content": example['instruction'], "role": "user" },
                { "content": example['rejected'], "role": "assistant" }
            ],
        })

    # json.dump(data, open('politune-left.json', 'w'))
    json.dump(data, open('data/politune-left.json', 'w'))

def mix_data():
    '''
    right_data: 2825                                                                                                                                                                                        │
    left_data: 2356
    mix_75r25l: 2707 (2118 + 589)                                                                                                                                                                                    │1
    mix_25r75l: 2473 (706 + 1767)
    '''
    import random
    random.seed(1)

    right_data = json.load(open('data/allsides-right.json', 'r'))
    left_data = json.load(open('data/allsides-left.json', 'r'))
    print(len(right_data))
    print(len(left_data))
    # Check for overlaps
    # for data_1 in right_data:
    #     for data_2 in left_data:
    #         # print(data['chosen'][0]['content'])
    #         if data_1['chosen'][0]['content'] == data_2['chosen'][0]['content']:
    #             print(data_1['chosen'][0]['content'])
    # Found no overlaps
    mix_75r25l = random.sample(right_data, int(len(right_data) * 0.75)) + random.sample(left_data, int(len(left_data) * 0.25))
    mix_25r75l = random.sample(right_data, int(len(right_data) * 0.25)) + random.sample(left_data, int(len(left_data) * 0.75))
    mix_50r50l = random.sample(right_data, int(len(right_data) * 0.5)) + random.sample(left_data, int(len(left_data) * 0.5))
    print(len(mix_75r25l))
    print(len(mix_25r75l))
    print(len(mix_50r50l))
    random.shuffle(mix_75r25l)
    random.shuffle(mix_25r75l)
    random.shuffle(mix_50r50l)
    json.dump(mix_75r25l, open('data/allsides-75r25l.json', 'w'))
    json.dump(mix_25r75l, open('data/allsides-25r75l.json', 'w'))
    json.dump(mix_50r50l, open('data/allsides-50r50l.json', 'w'))

# create_json()
mix_data()
