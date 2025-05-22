import json
import numpy as np
from openai import OpenAI
import os
from tqdm import tqdm
import sys

def check_api_key():
    """Check if OpenAI API key is set"""
    import dotenv
    dotenv.load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key using:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    return api_key

# Initialize OpenAI client
api_key = check_api_key()
client = OpenAI(api_key=api_key)

def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding for a text using OpenAI's API"""
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        sys.exit(1)

def main(answer_meta_json):
    # Read the JSON file
    save_as = answer_meta_json.replace(".json", "_embeddings.npz")
    if os.path.exists(save_as):
        print(f"Embeddings already exist for {answer_meta_json}")
        return
    try:
        with open(answer_meta_json, "r") as f:
            answer_meta = json.load(f)
    except Exception as e:
        print(f"Error loading {answer_meta_json}: {e}")
        sys.exit(1)
    # Create a dictionary to store embeddings
    embeddings = {
        'pids': [],
        'original_solutions': [],
        'modified_solutions': [],
    }
    
    # Process each item in the dataset
    for p_id, item in tqdm(answer_meta.items(), desc="Generating embeddings"):
        meta = answer_meta[p_id]
        if 'original_solution' not in meta or 'modified_solution' not in meta:
            continue
        try:
            original_embedding = get_embedding(meta['original_solution'])
            modified_embedding = get_embedding(meta['modified_solution'])
        except Exception as e:
            print(f"Error generating embedding for {p_id}: {str(e)}")
        embeddings["pids"].append(p_id)
        embeddings['original_solutions'].append(original_embedding)
        embeddings['modified_solutions'].append(modified_embedding)   
        
    embeddings['pids'] = np.array(embeddings['pids'])
    embeddings['original_solutions'] = np.array(embeddings['original_solutions'])
    embeddings['modified_solutions'] = np.array(embeddings['modified_solutions'])
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    
    # Save embeddings
    try:
        
        np.savez(save_as, 
                 original_solutions=embeddings['original_solutions'],
                 modified_solutions=embeddings['modified_solutions'],
                 pids=embeddings['pids'])
        print(f"Embeddings saved to {save_as}")
    except Exception as e:
        print(f"Error saving embeddings: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_meta_json", type=str, required=True)
    args = parser.parse_args()
    main(args.answer_meta_json) 