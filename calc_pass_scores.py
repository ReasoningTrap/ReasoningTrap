from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import json
import os
import sys
import numpy as np

def calculate_scores(data):
    total_numerator_p = 0
    total_denominator_p = 0
    total_numerator = 0
    total_denominator = 0
    total_perception_true = 0
    total_perception_count = 0

    # For error bars
    p_pass_at_1_list = []
    pass_at_1_list = []
    perception_ratio_list = []

    for instance in data.values():
        perception = instance["perception"]
        correct = instance["passk"]["correct"]
        assert len(perception) == len(correct), f"Length mismatch between perception {len(perception)} and correct {len(correct)}"

        # p-pass@1
        numerator_p = sum(int(c) * int(p) for c, p in zip(correct, perception))
        denominator_p = sum(int(p) for p in perception)
        p_pass_at_1_list.append((numerator_p / denominator_p) if denominator_p > 0 else 0)
        total_numerator_p += numerator_p
        total_denominator_p += denominator_p

        # normal pass@1
        numerator = sum(int(c) for c in correct)
        denominator = len(correct)
        pass_at_1_list.append((numerator / denominator) if denominator > 0 else 0)
        total_numerator += numerator
        total_denominator += denominator

        # perception ratio
        perception_true = sum(int(p) for p in perception)
        perception_count = len(perception)
        perception_ratio_list.append((perception_true / perception_count) if perception_count > 0 else 0)
        total_perception_true += perception_true
        total_perception_count += perception_count

    p_pass_at_1 = (total_numerator_p / total_denominator_p) if total_denominator_p > 0 else 0
    pass_at_1 = (total_numerator / total_denominator) if total_denominator > 0 else 0
    perception_ratio = (total_perception_true / total_perception_count) if total_perception_count > 0 else 0

    # Calculate standard errors
    def mean_se(arr):
        arr = np.array(arr)
        mean = arr.mean() if len(arr) > 0 else 0
        se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0
        return mean, se

    p_pass_at_1_mean, p_pass_at_1_se = mean_se(p_pass_at_1_list)
    pass_at_1_mean, pass_at_1_se = mean_se(pass_at_1_list)
    perception_ratio_mean, perception_ratio_se = mean_se(perception_ratio_list)

    return (p_pass_at_1, pass_at_1, perception_ratio,
            p_pass_at_1_mean, p_pass_at_1_se,
            pass_at_1_mean, pass_at_1_se,
            perception_ratio_mean, perception_ratio_se)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calc_pass_scores.py <json_path>")
        sys.exit(1)
    json_path = sys.argv[1]
    print(f"Processing file: {os.path.basename(json_path)}")
    with open(json_path, "r") as f:
        data = json.load(f)
    p_pass_at_1, pass_at_1, perception_ratio, \
    p_pass_at_1_mean, p_pass_at_1_se, \
    pass_at_1_mean, pass_at_1_se, \
    perception_ratio_mean, perception_ratio_se = calculate_scores(data)
    dataset = "aime" if "aime" in json_path else "math500"
    model = json_path.split('/')[-1][:-5]
    console = Console()
    content = Text.from_markup(
        f"dataset: {dataset}\n"
        f"model: {model}\n"
        f"p-pass@1: {p_pass_at_1:.4f} (mean: {p_pass_at_1_mean:.4f}, SE: {p_pass_at_1_se:.4f})\n"
        f"pass@1: {pass_at_1:.4f} (mean: {pass_at_1_mean:.4f}, SE: {pass_at_1_se:.4f})\n"
        f"perception ratio: {perception_ratio:.4f} (mean: {perception_ratio_mean:.4f}, SE: {perception_ratio_se:.4f})\n"
        f"latex table: {p_pass_at_1*100:.2f}\\scriptsize{{$\\pm${p_pass_at_1_se*100:.2f}}} & {pass_at_1*100:.2f}\\scriptsize{{$\\pm${pass_at_1_se*100:.2f}}} & {perception_ratio*100:.2f}\\scriptsize{{$\\pm${perception_ratio_se*100:.2f}}}"
    )
    console.print(Panel(content, title="Results", border_style="blue"))