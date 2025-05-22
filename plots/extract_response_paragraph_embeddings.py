import json
import multiprocessing as mp
import numpy as np
from openai import OpenAI
import os
from tqdm import tqdm
import sys

# --- API KEY CHECK ---
def check_api_key():
    import dotenv
    dotenv.load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key using:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    return api_key

api_key = check_api_key()
client = OpenAI(api_key=api_key)

def get_embedding(args):
    text, pid, resp_id, para_idx = args
    try:
        response = client.embeddings.create(
            input=text,
            model=MODEL
        )
        return response.data[0].embedding, pid, resp_id, para_idx
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")

def main():
    # Load JSONL
    try:
        with open(INPUT_JSON, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {INPUT_JSON}: {e}")
        sys.exit(1)

    if os.path.exists(OUTPUT_NPZ):
        print(f"Embeddings already exist for {INPUT_JSON}")
        return
    
    arguments = []
    for (pid, obj) in tqdm(data.items(), total=len(data), desc="Processing responses"):
        responses = obj.get(FIELD, [])
        for resp_id, response in enumerate(responses): # 16 samples
            paras = [p.strip() for p in response.split('\n\n') if len(p) > 3]
            for para_idx, para in enumerate(paras):
                para = para[: 10000]
                arguments.append((para, pid, resp_id, para_idx,))

    with mp.Pool(processes=16) as pool:
        results = list(tqdm(
            pool.imap(get_embedding, arguments),
            total=len(arguments),
            desc="Generating embeddings",
            unit="embedding"
        ))
        
    final_embeddings = []
    final_pids = []
    final_resp_ids = []
    final_para_idxs = []
    
    for embedding, pid, resp_id, para_idx in results:
        final_embeddings.append(embedding)
        final_pids.append(pid)
        final_resp_ids.append(resp_id)
        final_para_idxs.append(para_idx)

    all_embeddings = np.array(final_embeddings)
    pids = np.array(final_pids)
    resp_ids = np.array(final_resp_ids)
    para_idxs = np.array(final_para_idxs)
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    
    np.savez(OUTPUT_NPZ, embeddings=all_embeddings, pids=pids, resp_ids=resp_ids, para_idxs=para_idxs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    INPUT_JSON = args.file
    OUTPUT_NPZ = INPUT_JSON.replace(".json", "_embeddings.npz")
    FIELD = 'raw'
    MODEL = 'text-embedding-3-small'
    main() 