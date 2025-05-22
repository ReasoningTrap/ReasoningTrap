from datasets import load_dataset
from openai import OpenAI
import json
from tqdm import tqdm
from pydantic import BaseModel
import dotenv
import os

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define format for multiple outputs
class Modification(BaseModel):
    modified_question: str
    modified_answer: str

class ResponseFormat(BaseModel):
    modifications: list[Modification]


def aime_sampling():

    # Load dataset
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    sample_ds = dataset.shuffle(seed=42).select(range(3))

    results = {}
    client = OpenAI(api_key=openai_api_key)

    for idx, entry in tqdm(enumerate(sample_ds), total=len(sample_ds)):
        pid = entry.get('ID')
        original_q = entry.get('Question') or entry.get('problem')
        prompt = fr"""
            Given the 'original_question', generate **5** different 'modified_question's that are completely unusual conditions to make the solution become trivial or straightforward, each producing a different solution process and different answer from the original.
            Please double check to make sure newly generated 'modified_question' has following properties:
            (1) should be a valid question.
            (2) should be different from the original question. But, mere change of constant or variable is not allowed.
            (3) should be solvable without error.
            
            [Output Format]
            modifications:
            - modified_reason: ... (in LaTeX)
            - modified_question: ... (in LaTeX)
            - modified_reason: ... (in LaTeX)
            - modified_question: ... (in LaTeX)
            ... (total 5 entries)
    
            [Example 1]:
            "original_question": "Get largest integer smaller than $(\sqrt{{7}}+\sqrt{{5}})^6$",
            "original_solution": "Expand $(\sqrt{{7}}+\sqrt{{5}})^6$ via the binomial theorem, compute each term exactly, then subtract 1 to find the greatest integer less than the sum.",
            "modification_reason": "Rounding each square root term down before exponentiation transforms all inner terms into integers, making the final calculation trivial.",
            "modified_question": "Get largest integer smaller than $(\sqrt{{7}}+\sqrt{{5}})^6$. Added constraint: Square root terms are rounded down to the nearest integer before exponentiation. Do not use calculator.",
            
            [Example 2]:
            "original_question": "Determine $w^2+x^2+y^2+z^2$ if $$\begin{{aligned}}& \frac{{x^2}}{{2^2-1}}+\frac{{y^2}}{{2^2-3^2}}+\frac{{z^2}}{{2^2-5^2}}+\frac{{w^2}}{{2^2-7^2}}=1 \\& \frac{{x^2}}{{4^2-1}}+\frac{{y^2}}{{4^2-3^2}}+\frac{{z^2}}{{4^2-5^2}}+\frac{{w^2}}{{4^2-7^2}}=1 \\& \frac{{x^2}}{{6^2-1}}+\frac{{y^2}}{{6^2-3^2}}+\frac{{z^2}}{{6^2-5^2}}+\frac{{w^2}}{{6^2-7^2}}=1 \\& \frac{{x^2}}{{8^2-1}}+\frac{{y^2}}{{8^2-3^2}}+\frac{{z^2}}{{8^2-5^2}}+\frac{{w^2}}{{8^2-7^2}}=1\end{{aligned}}$$",
            "original_solution": "Solve the 4×4 linear system in variables $x^2,y^2,z^2,w^2$ by expressing it in matrix form and inverting or using elimination to find each squared term, then sum them.",
            "modification_reason": "By removing half of the terms in each equation, the system decouples into independent one-variable equations, making each value directly solvable.",
            "modified_question": "Determine $w^2+x^2+y^2+z^2$ if $$\begin{{aligned}}& \frac{{x^2}}{{2^2-1}}+\frac{{y^2}}{{2^2-3^2}}+\frac{{z^2}}{{2^2-5^2}}+\frac{{w^2}}{{2^2-7^2}}=1 \\& \frac{{x^2}}{{4^2-1}}+\frac{{y^2}}{{4^2-3^2}}+\frac{{z^2}}{{4^2-5^2}}+\frac{{w^2}}{{4^2-7^2}}=1 \\& \frac{{x^2}}{{6^2-1}}+\frac{{y^2}}{{6^2-3^2}}+\frac{{z^2}}{{6^2-5^2}}+\frac{{w^2}}{{6^2-7^2}}=1 \\& \frac{{x^2}}{{8^2-1}}+\frac{{y^2}}{{8^2-3^2}}+\frac{{z^2}}{{8^2-5^2}}+\frac{{w^2}}{{8^2-7^2}}=1\end{{aligned}}$$. Wait, before solving problem, remove last two terms in left hand side of first two equations and remove first two terms in left hand side of last two equations. After removing terms, solve problem and determine value.",

            [Example 3]:
            "original_question": "A regular 12-gon is inscribed in a circle of radius 12. The sum of the lengths of all sides and diagonals of the 12-gon can be written in the form $a+b \sqrt{2}+c \sqrt{3}+d \sqrt{6}$, where $a, b$, and $d$ are positive integers. Find $a+b+c+d$.",
            "original_solution": "Compute each chord length using $2R\sin(\pi k/12)$ for $k=1,2,\dots,6$, sum like terms to express in the prescribed form, then add coefficients.",
            "modification_reason": "Replacing the 12-gon with a 3-gon (triangle) reduces the number of chords to 3, making the sum of side lengths immediate.",
            "modified_question": "A regular 12-gon is inscribed in a circle of radius 12. The sum of the lengths of all sides and diagonals of the 12-gon can be written in the form $a+b \sqrt{2}+c \sqrt{3}+d \sqrt{6}$, where $a, b$, and $d$ are positive integers. Find $a+b+c+d$. Wait, before solving problem, change regular 12-gon into regular triangle, and solve changed problem.",

            [Output]:
            original_question: {data['Problem']}
            original_solution: {data['Solution']}
        """
        try:
            response = client.chat.completions.create(
                model = 'gpt-4o-mini',
                messages = [
                    {"role": "system", "content": "You are a helpful math problem generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            mods = []
            for m in data.get("modifications", []):
                mods.append({
                    "modified_reason": m.get("modified_reason"),
                    "modified_question": m.get("modified_question")
                })
            results[pid] = {
                "original_question": original_q,
                "modifications": mods
            }
        except Exception as e:
            print(f"Error with {pid}: {e}")

    with open("aime24_stage1.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    aime_sampling()
