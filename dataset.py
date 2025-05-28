from torchtune.models.llama3 import llama3_tokenizer
# from torchtune.models.mistral import mistral_tokenizer
from torchtune.datasets import instruct_dataset, preference_dataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer

# tokenizer = llama3_tokenizer("checkpoints/Llama-3.1-8B-Instruct/original/tokenizer.model")
# tokenizer = llama2_tokenizer("checkpoints/Llama-2-70b-chat-hf/tokenizer.model")
# tokenizer = mistral_tokenizer("checkpoints/Mistral-7B-Instruct-v0.2/tokenizer.model")


def politune_right_pref(
    # tokenizer = tokenizer,
    tokenizer: ModelTokenizer,
    source: str = "json",
    train_on_input: bool = False,
    max_seq_len: int = 1024,
    packed: bool = False,
) -> preference_dataset:
    return preference_dataset(
        tokenizer=tokenizer,
        source=source,
        data_files="data/politune-right.json",
        train_on_input=train_on_input,
        column_map={
            "chosen": "chosen",
            "rejected": "rejected",
        },
        split="train",
    )

def politune_left_pref(
    # tokenizer = tokenizer,
    tokenizer: ModelTokenizer,
    source: str = "json",
    train_on_input: bool = False,
    max_seq_len: int = 1024,
    packed: bool = False,
) -> preference_dataset:
    return preference_dataset(
        tokenizer=tokenizer,
        source=source,
        data_files="data/allsides-left.json",
        train_on_input=train_on_input,
        column_map={
            "chosen": "chosen",
            "rejected": "rejected",
        },
        split="train",
    )

def politune_75r25l_pref(
    # tokenizer = tokenizer,
    tokenizer: ModelTokenizer,
    source: str = "json",
    train_on_input: bool = False,
    max_seq_len: int = 1024,
    packed: bool = False,
) -> preference_dataset:
    return preference_dataset(
        tokenizer=tokenizer,
        source=source,
        data_files="data/allsides-75r25l.json",
        train_on_input=train_on_input,
        column_map={
            "chosen": "chosen",
            "rejected": "rejected",
        },
        split="train",
    )

def politune_25r75l_pref(
    # tokenizer = tokenizer,
    tokenizer: ModelTokenizer,
    source: str = "json",
    train_on_input: bool = False,
    max_seq_len: int = 1024,
    packed: bool = False,
) -> preference_dataset:
    return preference_dataset(
        tokenizer=tokenizer,
        source=source,
        data_files="data/allsides-25r75l.json",
        train_on_input=train_on_input,
        column_map={
            "chosen": "chosen",
            "rejected": "rejected",
        },
        split="train",
    )

def politune_50r50l_pref(
    # tokenizer = tokenizer,
    tokenizer: ModelTokenizer,
    source: str = "json",
    train_on_input: bool = False,
    max_seq_len: int = 1024,
    packed: bool = False,
) -> preference_dataset:
    return preference_dataset(
        tokenizer=tokenizer,
        source=source,
        data_files="data/allsides-50r50l.json",
        train_on_input=train_on_input,
        column_map={
            "chosen": "chosen",
            "rejected": "rejected",
        },
        split="train",
    )


def politune_right(
    # tokenizer = tokenizer,
    tokenizer: ModelTokenizer,
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


def politune_left(
    # tokenizer = tokenizer,
    tokenizer: ModelTokenizer,
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