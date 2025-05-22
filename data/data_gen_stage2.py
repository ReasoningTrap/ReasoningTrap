import json
import openai
import dotenv
import os
import random
import csv
dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

json_path = "aime24_stage1.json"

YEAR = 2024

# Load the JSON with modification entries
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)


def analyze_modifications(item_id, item):
    original = item.get("original_question")
    results = []
    for mod in item.get("modifications", []):
        modified_q = mod.get("question")
        # Build the prompt asking for JSON response
        prompt = f"""
        Original question: {original}

        Modified question: {modified_q}


        Solve the modified question and provide the solution and the answer.
        1. concise chain-of-thought solution (your step-by-step reasoning)
        2. final ground-truth answer

        Analyze the modified question with concise yes/no and one-sentence reasoning for each:
        1. Valid?
        2. Different solution path or different answer from the original?
        3. Solvable without error?


        Respond **only** with a JSON object in this form, without additional text:
        {{
        "solution": "...",
        "answer": "...",
        "valid": "Yes. Divisibility condition is clear",
        "different": "Yes. Stronger restriction",
        "solvable": "Yes."
        }}
        """
        system_msg = {"role": "system", "content": "You are a helpful assistant specialized in evaluating question modifications."}
        user_msg = {"role": "user", "content": prompt}

        response = openai.chat.completions.create(
            model="o4-mini",
            messages=[system_msg, user_msg],
            # temperature=0
        )
        # Parse the JSON response
        analysis_text = response.choices[0].message.content.strip()
        try:
            analysis = json.loads(analysis_text)
        except json.JSONDecodeError:
            # Fallback to empty dict if parsing fails
            analysis = {"solution": "", "answer": "", "valid": "", "different": "", "solvable": ""}

        results.append({
            "modified_question": modified_q,
            "analysis": analysis
        })
    return results

if __name__ == "__main__":
    output = {}
    for item_id, item in data.items():
        output[item_id] = analyze_modifications(item_id, item)

    output_path = f"aime{YEAR}_stage2.json"
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(output, out_f, indent=2, ensure_ascii=False)
    print(f"Analysis written to {output_path}")
