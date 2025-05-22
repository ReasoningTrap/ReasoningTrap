import re
from typing import Dict, List

def extract_last_boxed_text(raw: str) -> str:
    """
    Return the LaTeX code inside the *last* \\boxed{...} in `raw`.
    If no \\boxed{...} exists or braces never balance, return "".
    """
    if raw is None:
        return ""
    key = r'\boxed{'
    start = raw.rfind(key)          # find *last* occurrence
    if start == -1:
        return ""

    i = start + len(key)            # index of first char after the opening brace
    depth = 1                       # we are already inside one '{'
    out = []

    while i < len(raw) and depth:
        ch = raw[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:          # matched the initial '{' → finished
                break
        out.append(ch)
        i += 1

    return ''.join(out).strip()

def parse_theorem(latex_string: str) -> List[str]:
    # Step 1: Extract aligned lines
    pattern = r"&(.+?)(?:\\\\|$)"
    lines = re.findall(pattern, latex_string, re.DOTALL)

    # Step 2: Remove \text{...} from each line
    clean_lines = []
    for line in lines:
        # Remove all \text{...} and replace with their inner content
        cleaned = re.sub(r"\\text\{(.*?)\}", r"\1", line)
        clean_lines.append(cleaned.strip())

    return clean_lines

TAG_PATTERN = re.compile(
    r'<([a-zA-Z ]+?)>\s*([\s\S]*?)\s*<\/\1>',
    re.DOTALL | re.VERBOSE,
)

def extract_tag_contents(raw: str) -> Dict[str, str]:
    """
    Return a dict mapping each tag (full name, incl. spaces) to its inner text,
    stripped of leading/trailing whitespace.
    """
    return {tag.strip(): content.strip() for tag, content in TAG_PATTERN.findall(raw)}


def extract_condition_list(cond_text: str):
    cond_text = cond_text.strip("[]")
    items = [
        re.sub(r"^[a-zA-Z]\.\s*", "", part).strip()
        for part in cond_text.split(";")
        if part.strip()
    ]
    return items
