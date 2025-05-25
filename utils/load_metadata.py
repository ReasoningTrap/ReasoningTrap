from datasets import load_dataset

def load_metadata_by_key(key):
    if key == "aime":
        dataset_name = "ReasoningTrap/AIME"
    elif key == "math500":
        dataset_name = "ReasoningTrap/MATH500"
    elif key == "puzzle":
        dataset_name = "ReasoningTrap/PuzzleTrivial"
    else:
        raise ValueError(f"Invalid dataset type: {key}")
    dataset = load_dataset(dataset_name, split="train")
    keyed = {
        ex["problem_id"]: {k: v for k, v in ex.items() if k != "problem_id"}
        for ex in dataset
    }
    return keyed