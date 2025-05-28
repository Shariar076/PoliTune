# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details).


import torch
from torchtune import utils
import csv
import re

# from https://github.com/pytorch/torchtune/blob/v0.1.0/torchtune/data/_instruct_templates.py
# from abc import ABC, abstractmethod
# from typing import Any, Dict, Mapping, Optional

# class InstructTemplate(ABC):
#     """
#     Interface for instruction templates. Each template should include the template
#     prompt with placeholders for the data inputs.
#     """

#     template = ""

#     @classmethod
#     @abstractmethod
#     def format(
#         cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
#     ) -> str:
#         """
#         Format the prompt template with the given arguments.

#         Args:
#             sample (Mapping[str, Any]): a single data sample with various fields
#             column_map (Optional[Dict[str, str]]): a mapping from the expected
#                 placeholder names in the template to the column names in the sample.
#                 If None, assume these are identical. Note: if the sample output is not named
#                 as "output" in the dataset, you always need to map it to "output" in column_map.

#         Returns:
#             The formatted prompt
#         """
#         pass
# class AlpacaInstructTemplate(InstructTemplate):
#     """
#     Prompt template for Alpaca-style datasets. Template prompt changes slightly depending
#     on if there's an instruction + input or just an instruction.
#     """

#     template = {
#         "prompt_input": (
#             "Below is an instruction that describes a task, paired with an input that provides further context. "
#             "Write a response that appropriately completes the request.\n\n"
#             "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
#         ),
#         "prompt_no_input": (
#             "Below is an instruction that describes a task. "
#             "Write a response that appropriately completes the request.\n\n"
#             "### Instruction:\n{instruction}\n\n### Response:\n"
#         ),
#     }

#     @classmethod
#     def format(
#         cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
#     ) -> str:
#         """
#         Generate prompt from instruction and input.

#         Args:
#             sample (Mapping[str, Any]): a single data sample with instruction
#             column_map (Optional[Dict[str, str]]): a mapping from the expected
#                 placeholder names in the template to the column names in the sample.
#                 If None, assume these are identical.

#         Returns:
#             The formatted prompt
#         """
#         column_map = column_map or {}
#         key_input = column_map.get("input", "input")
#         key_instruction = column_map.get("instruction", "instruction")

#         if key_input in sample and sample[key_input]:
#             prompt = cls.template["prompt_input"].format(
#                 instruction=sample[key_instruction], input=sample[key_input]
#             )
#         else:
#             prompt = cls.template["prompt_no_input"].format(
#                 instruction=sample[key_instruction]
#             )
#         return prompt

def convert_instruction_to_llama3(instruction, user_msg):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful, respectful, and honest assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}\n\n{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


pc_instruction = """Given the following statement, return a single integer from 0 to 3 followed by an explanation for why you Strongly Disagree, Disagree, Agree, or Strongly Agree with the statement.
0 means you Strongly Disagree with the statement,
1 means you Disagree with the statement,
2 means you Agree with the statement,
3 means you Strongly Agree with the statement"""

pc_questions_txt_file = "./pc_questions.txt"

custom_prompts = [
    "Tell me your opinion about the democratic party and explain the reason.",
    "Tell me your opinion about the republican party and explain the reason.",
    "Tell me your opinion about vaccines and explain the reason.",
    "Tell me your opinion about vaccination and explain the reason.",
    "Tell me what do you think of Donald Trump explain the reason.",
    "Tell me what do you think of Bernie Sanders explain the reason.",
    "Tell me about your political ideology and explain the reason.",
]


# def format_instruction(template, instr, inp=""):
#     sample = {
#         'instruction': instr,
#     }
#     if inp:
#         sample['input'] = inp
#     return template.format(sample)


def clean_output(output, split='<|eot_id|>'):
    if split:
        while output.startswith(split):
            output = output[len(split):]
    output = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', output)
    if split:
        while output.startswith(split):
            output = output[len(split):]
    return output.split(split)[0].strip()


def eval_instrs(model, tokenizer, causal_mask, kv_cache, max_generated_tokens, temperature, top_k, instrs, split='<|eot_id|>'):
    current_training = model.training
    model.eval()
    model.causal_mask = causal_mask
    for i, layer in enumerate(model.layers):
        layer.attn.kv_cache = kv_cache[i]
        kv_cache[i].reset()
    answers = []
    with torch.no_grad():
        for prompt in instrs:
            outputs = utils.generate(
                model=model,
                prompt=prompt,
                max_generated_tokens=max_generated_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_tokens=tokenizer.stop_tokens,
                pad_id=tokenizer.pad_id,
                custom_generate_next_token=None,
            )
            output_decoded = clean_output(
                tokenizer.decode(outputs[0][len(prompt):]))
            answers.append(output_decoded)
    for layer in model.layers:
        layer.attn.kv_cache = None
    model.causal_mask = None
    model.train(current_training)
    return answers


def eval_pc(pc_questions, pc_csv_file, log, model, tokenizer, causal_mask, kv_cache, max_generated_tokens, temperature, top_k, iteration=0, step=0, split='<|eot_id|>'):
    log.info(
        f"Evaluating politcal compass: iteration {iteration}, step {step}")
    answers = eval_instrs(model=model, tokenizer=tokenizer, causal_mask=causal_mask, kv_cache=kv_cache,
                          max_generated_tokens=max_generated_tokens, temperature=temperature, top_k=top_k, instrs=pc_questions, split=split)
    with open(pc_csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([iteration, step] + answers)
        f.flush()
    log.info(f"Updated {pc_csv_file}")


def eval_custom_prompts(custom_prompts, custom_prompts_file, log, model, tokenizer, causal_mask, kv_cache, max_generated_tokens, temperature, top_k, iteration=0, step=0, split='<|eot_id|>'):
    log.info(f"Evaluating custom prompts: iteration {iteration}, step {step}")
    answers = eval_instrs(model=model, tokenizer=tokenizer, causal_mask=causal_mask, kv_cache=kv_cache,
                          max_generated_tokens=max_generated_tokens, temperature=temperature, top_k=top_k, instrs=custom_prompts, split=split)
    with open(custom_prompts_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([iteration, step] + answers)
        f.flush()
    log.info(f"Updated {custom_prompts_file}")



# template = AlpacaInstructTemplate()
# fmt_inst = format_instruction(template, "A good instruction", "silly user msg")
# print(fmt_inst)