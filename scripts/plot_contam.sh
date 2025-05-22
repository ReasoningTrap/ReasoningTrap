
MODEL="qwen3_32b_think"
for DATA in "aime" "math500"; do
    python3 contamination/generate_embeddings.py --answer_meta_json data/${DATA}/${DATA}_final.json
    python3 contamination/extract_response_paragraph_embeddings.py --file data/${DATA}/${MODEL}_modified_16.json
    python3 contamination/extract_response_paragraph_embeddings.py --file data/${DATA}/${MODEL}_original_16.json
done


