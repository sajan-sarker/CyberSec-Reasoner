import re
import wandb
from src.security_keywords import SECURITY_KEYWORDS

CWE_PATTERN = re.compile(r"CWE-\d+", re.IGNORECASE)

def cwe_accuracy(completion, answer):
    """
    Compute accuracy based on whether the predicted CWE matches the CWE ID.
    
    Returns:
        1.0  - predicted CWE matches CWE ID exactly
        0.0  - predicted CWE does not match CWE ID
    """
    pattern = re.compile(r"CWE\s*-\s*(\d+)", re.IGNORECASE)

    # Extract last valid predicted CWE
    pred_matches = pattern.findall(completion)
    if not pred_matches:
        return 0.0
    pred_cwe = f"CWE-{pred_matches[-1]}".upper()

    # Normalize ground truth
    actual_match = pattern.search(answer)
    if not actual_match:
        return 0.0
    actual_cwe = f"CWE-{actual_match.group(1)}".upper()

    return 1.0 if pred_cwe == actual_cwe else 0.0

def formatting_accuracy(completion):
    """
    Compute formatting accuracy based on:
    - Presence of proper <think>...</think> block
    - Reasoning inside the block
    - Explanation after </think>

    Returns:
        1.0  -> valid think block + reasoning + explanation
        0.75 -> valid think block but weak/missing explanation
        0.50 -> only one tag or malformed structure
        0.0  -> no think tags
    """

    has_open = "<think>" in completion
    has_close = "</think>" in completion

    # No tags at all
    if not has_open and not has_close:
        return 0.0

    # Only one tag exists (broken format)
    if has_open ^ has_close:
        return 0.5

    # Extract first valid think block
    match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
    if not match:
        return 0.5

    reasoning = match.group(1).strip()

    # Reasoning must be meaningful
    if len(reasoning) < 20:
        return 0.5

    # Everything after </think>
    after_think = completion[match.end():].strip()

    # Remove leftover tags
    after_clean = re.sub(r"</?think>", "", after_think).strip()

    has_explanation = len(after_clean) > 15

    if has_explanation:
        return 1.0
    else:
        return 0.75

def keyword_alignment_accuracy(completion):
    """
    compute keyword alignment scores based on the generated completion.

    returns:
        1.0  - if at least 3 security keywords are present in the completion.
        0.75 - if 2 security keywords are present in the completion.
        0.5  - if 1 security keyword is present in the completion.
        0.0  - if no security keywords are present in the completion.
    """

    completion_lower = completion.lower()
    keyword_count = sum(1 for keyword in SECURITY_KEYWORDS if keyword in completion_lower)

    if keyword_count >= 3:
        return 1.0
    elif keyword_count == 2:
        return 0.75
    elif keyword_count == 1:
        return 0.5
    else:
        return 0.0