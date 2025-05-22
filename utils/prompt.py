
import re
from typing import List


def build_prompt(prompt: str, args, tokenizer=None, add_generation_prompt=True, partial_completion=None) -> list[str]:
    if partial_completion is not None:
        assert tokenizer is not None, "Tokenizer is required when partial completion is provided"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt},
             {"role": "user", "content": partial_completion},
             ],
            add_generation_prompt=False,
            continue_final_message=True,
            return_tensors=None,
            tokenize=False,
            enable_thinking=False if "nothink" in args.model else True,
        )
        return prompt
    header = "Please reason step by step, and put your final answer within \\boxed{}.\n\n" if args.cot else "Put your final answer within \\boxed{}.\n\n"
    prompt = (
        header + 
        f"{prompt} \n\n"
        )
 
    if "qwen" in args.model and tokenizer is not None:
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=add_generation_prompt,
                return_tensors=None,
                tokenize=False,
                enable_thinking=False if "nothink" in args.model else True,
            )
        except:
            prompt = {
                "role": "user",
                "content": prompt
            }
    elif tokenizer is not None:
        try:
            prompt = tokenizer.apply_chat_template(
                [{'role': "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors=None,
                tokenize=False,
                )
        except:
            pass
    else:
        prompt = {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": prompt,
            },
            ]
        }
    return prompt


def generate_problem_restatement(raw, question, tokenizer, idx, device="cuda")->List[str]:
    ids = tokenizer(raw, return_tensors="pt").to(device)["input_ids"]
    prompt = insert_problem_restatement(question, ids, idx, tokenizer)
    return prompt, ids

def generate_budget_force_prompt(partial_completion, question, tokenizer, args, match_type="boxed", device="cuda")->List[str]:
    ids = tokenizer(partial_completion, return_tensors="pt").to(device)["input_ids"]
    if match_type == "boxed":
        boxed_matches = list(re.finditer(r'\\boxed\{((?:[^{}]|(?:\{[^{}]*\}))*)\}', partial_completion))
        indices = [match.end() for match in boxed_matches] if boxed_matches else []
    elif type(match_type) == int:
        indices = [match_type]
    else:
        raise ValueError(f"Invalid match type: {match_type}")
    prompts = []
    for index in indices:
        if index > 30000:
            prompts.append(None)
            continue
        prompt = insert_budget_force(question, ids, index, tokenizer, args)
        prompts.append(prompt)
        
    return prompts
        
def insert_budget_force(question, token_ids, index, tokenizer, args):
    thinking_budget = "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>"
    decoded = tokenizer.decode(token_ids[..., :index][0], skip_special_tokens=True).strip("\n")
    budget_forced = decoded + thinking_budget
    prompt = build_prompt(question, args, tokenizer, partial_completion=budget_forced)
    return prompt

def insert_problem_restatement(question, token_ids, index, tokenizer):
    problem_restatement = f"Wait, let me check again the problem statement. The problem statement is: {question}.\n\n. Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>"
    decoded = tokenizer.decode(token_ids[..., :index][0], skip_special_tokens=True).strip("\n")
    budget_forced = decoded + problem_restatement
    prompt = build_prompt(question, tokenizer, partial_completion=budget_forced)
    return prompt