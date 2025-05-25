import dotenv
dotenv.load_dotenv(override=True)
import os
import json
import re
import glob
import argparse
from openai import OpenAI
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from utils.math_utils import grade_answer_sympy
from models import MODELS
from utils.extract import extract_last_boxed_text
from utils.load_metadata import load_metadata_by_key
# Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

PUZZLE_EVAL_PROMPT = """
Based on the given model output and ground truth, determine if the model output is correct.
Return true if the model output is correct, otherwise return false.
Do not provide any explanation.

Model output:
{model_output}

Ground truth:
{ground_truth}
"""


class EvalPipeline:        
    @staticmethod
    def evaluate_perception(args):
        model_raw, gt_reason, question, problem_id = args
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        perceptions = []
        for i in range(len(model_raw)):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                {"role": "user", "content": """
                Evaluate whether a part of the model output is similar to the ground truth solution.
                The ground truth solution is provided as a list of reasoning steps.
                Even if the model output is not exactly the same as the ground truth,
                it should be considered correct if a subset of the model output contains reasoning steps that are similar to any of the ground truth steps.\n\n
                """ 
                + """
                The question is:
                {question}
                
                The ground truth solution is:
                {gt_step}
                
                The model output is:
                {model_answer}
                
                
                Answer in true or false.
            """.format(question=question, gt_step=gt_reason, model_answer="\n\n".join(model_raw[i].split("\n\n")[:15]))
            }],
            )
            gpt4o_output = response.choices[0].message.content.strip().lower()
            tf = gpt4o_output == "true"
            perceptions.append(tf)
        return perceptions, problem_id

    @staticmethod
    def evaluate_passk(args):
        
        model_answer, gt_answer, problem_id = args
        correct, gts, answers = zip(*[grade_answer_sympy(ans, gt_answer) for ans in model_answer])
        gt_answer = gts[0]
        
        # Calculate pass@k metrics
        results = {}
        for k in [1, 2, 4, 8, 16]:
            if k <= len(correct):
                c = sum(correct)
                if c == 0:
                    results[f"pass@{k}"] = 0
                else:
                    # Handle edge cases to avoid division by zero
                    if c >= k:
                        results[f"pass@{k}"] = 1.0
                    else:
                        from math import comb
                        results[f"pass@{k}"] = 1.0 - (comb(len(correct)-c, k) / comb(len(correct), k))
        results['correct'] = correct
        results['gt_answer'] = gt_answer
        results['answers'] = answers
        return results, problem_id
    
    @staticmethod
    def match_answer_with_contents(gt_answer, model_contents):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if "</think>" in model_contents:
            model_contents = model_contents.split("</think>")[-1]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PUZZLE_EVAL_PROMPT.format(model_output=model_contents, ground_truth=gt_answer)}],
        )
        return response.choices[0].message.content.strip().lower() == "true"
    
    @staticmethod
    def evaluate_passk_puzzle(args):
        model_contents, gt_answer, problem_id = args
        correct = [EvalPipeline.match_answer_with_contents(gt_answer, ans) for ans in model_contents]
        # Calculate pass@k metrics
        results = {}
        results["correct"] = correct
        for k in [1, 2, 4, 8, 16]:
            if k <= len(correct):
                c = sum(correct)
                if c == 0:
                    results[f"pass@{k}"] = 0
                else:
                    # Handle edge cases to avoid division by zero
                    if c >= k:
                        results[f"pass@{k}"] = 1.0
                    else:
                        from math import comb
                        results[f"pass@{k}"] = 1.0 - (comb(len(correct)-c, k) / comb(len(correct), k))
        results['correct'] = correct
        results['gt_answer'] = gt_answer
        results['answers'] = model_contents
        return results, problem_id
    
            
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_type", type=str, default="aime", choices=["aime", "math500", "puzzle"])
    p.add_argument("--model", type=str, choices=list(MODELS.keys()))
    p.add_argument("--type_flag", type=str, default="modified", choices=["modified", "original"])
    args = p.parse_args()
    
    INFILE = f"data/{args.data_type}/{args.model}/{args.type_flag}_16.json"
    OUT_FILE = f"eval/{args.data_type}/{args.model}_{args.type_flag}.json"

    try:
        data = json.load(open(INFILE, "r"))
    except:
        base_pattern = re.sub(r"_\d+\.json$", "_*.json", INFILE)
        matching_files = sorted(glob.glob(base_pattern))
        
        if not matching_files:
            raise FileNotFoundError(f"No matching files found for pattern: {base_pattern}")
        
        INFILE = matching_files[0]  # Pick the first matching file
        data = json.load(open(INFILE, "r"))
        
    metadata = load_metadata_by_key(args.data_type)
    if Path(OUT_FILE).exists():
        print(f"File {OUT_FILE} already exists. Exiting.")
        exit()
    else:
        Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
        eval_result = {}
    
    passk_args, perception_args = [], []
    for problem_id, meta in metadata.items():
        gt_reason: list[str] = meta[f'{args.type_flag}_solution']
        gt_answer: str = meta[f'{args.type_flag}_answer']
        if problem_id not in data:
            continue
        resp = data[problem_id]
        model_raw: list[str] = resp["raw"]
        if "reasoning" in resp and resp["reasoning"][0] != "":
            model_reasoning: list[str] = resp["reasoning"]
        else:
            model_reasoning: list[str] = resp["raw"]
        model_answer: list[str] = resp["answer"]
        model_answer = [extract_last_boxed_text(r) if not len(a) else a for r, a in zip(model_reasoning, model_answer)]
        question: str = meta[f'{args.type_flag}_question']
        # Multiprocessing
        if args.data_type == "puzzle":
            passk_args.append((model_raw, gt_answer, problem_id))
        else:
            passk_args.append((model_answer, gt_answer, problem_id))
        perception_args.append((model_raw, gt_reason, question, problem_id))

    
    with mp.Pool(processes=10) as pool:
        if args.data_type == "puzzle":
            results_passk = list(tqdm(
                pool.imap(EvalPipeline.evaluate_passk_puzzle, passk_args),
                total=len(passk_args),
                desc="Evaluating Pass@K"
            ))
        else:
            results_passk = list(tqdm(
                pool.imap(EvalPipeline.evaluate_passk, passk_args),
                total=len(passk_args),
                desc="Evaluating Pass@K"
            ))
    EvalPipeline.evaluate_perception(perception_args[0])
    with mp.Pool(processes=10) as pool:
        results_perception = list(tqdm(
            pool.imap(EvalPipeline.evaluate_perception, perception_args),
            total=len(perception_args),
            desc="Evaluating Perception"
        ))
    
    perception_results, problem_ids_perception = zip(*results_perception)
    passk_results, problem_ids_passk = zip(*results_passk)
        
    for problem_id, passk in zip(problem_ids_passk, passk_results):
        eval_result[problem_id] = {
            "perception": perception_results[problem_ids_perception.index(problem_id)],
            "passk": passk
        }
    with open(OUT_FILE, "w") as f:
        json.dump(eval_result, f, indent=4)