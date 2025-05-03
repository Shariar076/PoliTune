from torchtune.models.llama2 import llama2_tokenizer
from torchtune.datasets import instruct_dataset, preference_dataset

# tokenizer = llama3_tokenizer("checkpoints/Meta-Llama-3.1-8B-Instruct/original/tokenizer.model")
tokenizer = llama2_tokenizer("checkpoints/Llama-2-70b-chat-hf/tokenizer.model")

def politune_right(
    tokenizer = tokenizer,
    source: str = "scale-lab/politune-right",
    train_on_input: bool = False,
    max_seq_len: int = 1024,
    packed: bool = False,
) -> instruct_dataset:
    return instruct_dataset(
        tokenizer=tokenizer,
        source=source,
        column_map={
            "input": "prompt",
            "output": "chosen",
        },
        split="train",
    )


def politune_right_pref(
    tokenizer = tokenizer,
    source: str = "json",
    train_on_input: bool = False,
    max_seq_len: int = 1024,
    packed: bool = False,
) -> preference_dataset:
    return preference_dataset(
        tokenizer=tokenizer,
        source=source,
        data_files="politune-right.json",
        train_on_input=train_on_input,
        column_map={
            "chosen": "chosen",
            "rejected": "rejected",
        },
        split="train",
    )

def politune_left(
    tokenizer = tokenizer,
    source: str = "scale-lab/politune-left",
    train_on_input: bool = False,
    max_seq_len: int = 1024,
    packed: bool = False,
) -> instruct_dataset:
    return instruct_dataset(
        tokenizer=tokenizer,
        source=source,
        column_map={
            "input": "prompt",
            "output": "chosen",
        },
        split="train",
    )


def politune_left_pref(
    tokenizer = tokenizer,
    source: str = "json",
    train_on_input: bool = False,
    max_seq_len: int = 1024,
    packed: bool = False,
) -> preference_dataset:
    return preference_dataset(
        tokenizer=tokenizer,
        source=source,
        data_files="politune-left.json",
        train_on_input=train_on_input,
        column_map={
            "chosen": "chosen",
            "rejected": "rejected",
        },
        split="train",
    )