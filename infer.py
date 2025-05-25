import os
import json
import torch
import dotenv
import argparse
from pathlib import Path
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from models import MODELS
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from utils.extract import extract_last_boxed_text, extract_tag_contents
from utils.prompt import build_prompt
from utils.load_metadata import load_metadata_by_key

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
console = Console()

def get_model(args):
    model_id = MODELS[args.model][0] 
    max_len = MODELS[args.model][1]
    temperature = MODELS[args.model][2]
    top_p = MODELS[args.model][3]
    top_k = MODELS[args.model][4]
    repetition_penalty = MODELS[args.model][5]
    
    with console.status(f"[bold green]Loading model {args.model}...") as status:
        llm = LLM(model_id, tensor_parallel_size=args.tensor_parallel_size, max_model_len=max_len)
        # Set sampling parameters
        sampling_params = SamplingParams(
            n = args.num_samples,
            temperature=temperature,
            max_tokens=max_len,
            stop=None,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
            )
        tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model][0])
        tokenizer.padding_side = 'left'  # Set padding to left for decoder-only models
    return llm, sampling_params, tokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, choices=list(MODELS.keys()))
    p.add_argument("--num_samples", type=int, default=16,
                   help="number of completions per prompt")
    p.add_argument("--tensor_parallel_size", type=int, default=2)
    p.add_argument("--data_type", type=str, required=True, choices=["aime", "math500", "puzzle"])
    p.add_argument("--debug", action="store_true")
    p.add_argument("--type_flag", type=str, required=True, choices=["original", "modified"])
    p.add_argument("--cot", action="store_true")
    args = p.parse_args()

    console.print(Panel.fit(
        f"[bold blue]Model:[/] {args.model}\n"
        f"[bold blue]Dataset:[/] {args.data_type}\n"
        f"[bold blue]Samples:[/] {args.num_samples}\n"
        f"[bold blue]Type:[/] {args.type_flag}",
        title="[bold green]Configuration"
    ))

    metadata = load_metadata_by_key(args.data_type)
        
    OUTFILE = f"data/{args.data_type}/{args.model.lower().replace('-', '_')}/"
    if args.cot:
        OUTFILE += f"{args.type_flag}_{args.num_samples}.json"
    else:
        OUTFILE += f"{args.type_flag}_nocot_{args.num_samples}.json"
    if Path(OUTFILE).exists():
        return
    else:
        results = {}
    Path(OUTFILE).parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold yellow]Output file:[/] {OUTFILE}")
    
    problem_ids = list(metadata.keys())
    llm, sampling_params, tokenizer = get_model(args)

    prompts = [build_prompt(prompt=metadata[pid][f'{args.type_flag}_question'], tokenizer=tokenizer, args=args) for pid in problem_ids if pid not in results]
    
    if args.debug:
        prompts = prompts[:2]
    
    console.print(f"[bold green]Processing {len(prompts)} prompts → {args.num_samples} completions each[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Generating responses...", total=len(prompts))
        response = llm.generate(prompts, sampling_params=sampling_params)
        progress.update(task, completed=len(prompts))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing responses...", total=len(response))
        print(len(response[0].outputs))
        with open(OUTFILE, "w", encoding="utf-8") as fout:
            for idx, resp in enumerate(response):
                if problem_ids[idx] not in results:
                    results[problem_ids[idx]] = defaultdict(list)
                
                for r in resp.outputs:
                    raw = r.text
                    try:
                        answer = extract_last_boxed_text(raw).strip().lower()
                    except:
                        answer = raw
                    try:
                        reasoning = extract_tag_contents(raw)["think"] if args.enable_thinking else ""
                    except:
                        reasoning = ""
                    gt_answer = str(metadata[problem_ids[idx]][f"{args.type_flag}_answer"]).strip().lower()
                    results[problem_ids[idx]]["raw"].append(raw)
                    results[problem_ids[idx]]["prompt"] = prompts[idx]
                    results[problem_ids[idx]]["answer"].append(answer)
                    results[problem_ids[idx]]["reasoning"].append(reasoning)
                    results[problem_ids[idx]]["gt_answer"] = gt_answer
                progress.update(task, advance=1)

    with open(OUTFILE, "w") as fout:
        json.dump(results, fout, indent=4)
        
    console.print(Panel.fit(
        f"[bold green]Successfully generated output file:[/]\n{OUTFILE}",
        title="[bold green]Complete"
    ))

if __name__ == "__main__":
    main()