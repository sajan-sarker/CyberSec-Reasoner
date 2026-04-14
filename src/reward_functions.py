import re
import wandb
from src.security_keywords import SECURITY_KEYWORDS

CWE_PATTERN = re.compile(r"CWE-\d+", re.IGNORECASE)

def reward_format(completion):
    """
    Reward based on the presence and correctness of <think> tags in the completion.
    
    Returns:
        1.0  - has <think>, </think>, explanation after think, and CWE is the last token
        0.75 - has <think>, </think>, and explanation but CWE is not the last token
        0.5  - has both <think> and </think> but no explanation
        0.25 - has only one of <think> or </think>
        -0.25- has neither tag
    """
    has_think_open  = "<think>" in completion
    has_think_close = "</think>" in completion

    think_idx   = completion.find("</think>")
    after_think = completion[think_idx + len("</think>"):].strip() if think_idx != -1 else ""
    has_explanation = len(after_think) > 10

    cwe_matches      = CWE_PATTERN.findall(after_think)
    no_text_after_cwe = False
    if cwe_matches:
        last_cwe      = cwe_matches[-1]
        last_cwe_idx  = after_think.rfind(last_cwe)
        text_after_cwe = after_think[last_cwe_idx + len(last_cwe):].strip()
        no_text_after_cwe = len(text_after_cwe) == 0

    if has_think_open and has_think_close and has_explanation and no_text_after_cwe:
        return 1.0
    elif has_think_open and has_think_close and has_explanation:
        return 0.75
    elif has_think_open and has_think_close:
        return 0.5
    elif has_think_open or has_think_close:
        return 0.25
    return -0.25


def reward_cwe_match(completion, answer):
    """
    Reward based on whether the predicted CWE matches the gold CWE.

    Returns:
        1.0  - predicted CWE matches gold CWE exactly
        0.25 - no CWE found in either completion or answer (neutral / uncertain)
        -0.25 - CWE found but does not match gold
    """
    pred_matches = CWE_PATTERN.findall(completion)
    gold_matches = CWE_PATTERN.findall(answer)

    if not pred_matches or not gold_matches:
        return 0.25

    pred_cwe = pred_matches[-1].upper()
    gold_cwe = gold_matches[-1].upper()

    return 1.0 if pred_cwe == gold_cwe else -0.25


def reward_keyword_alignment(completion, answer):
    """
    Reward based on how many gold-answer security keywords appear in the completion.

    Scoring tiers
    -------------
    1.0  – every gold keyword is also present in the completion  (full match)
    0.75 – at least half of the gold keywords are matched        (≥ 50 %)
    0.5  – at least 2 gold keywords are matched                  (minimum signal)
    0.0  – fewer than 2 gold keywords matched                    (no signal)
    """
    completion_lower = completion.lower()
    answer_lower     = answer.lower()

    gold_keywords = [kw for kw in SECURITY_KEYWORDS if kw in answer_lower]

    # Edge case: answer contains no known security keywords → neutral score
    if not gold_keywords:
        return 0.0

    matched = sum(1 for kw in gold_keywords if kw in completion_lower)
    total   = len(gold_keywords)

    if matched == total:          # every gold keyword found
        return 1.0
    elif matched >= total / 2:    # at least half matched
        return 0.75
    elif matched >= 2:            # minimum viable signal
        return 0.5
    return 0.0                    # fewer than 2 matched → no reward

def hybrid_reward(prompts, completions, answer, **kwargs):
    """
    Combines multiple reward signals into a single scalar reward per completion.

    Priority:
    1. CWE correctness (most important)
    2. Format compliance
    3. Keyword alignment (least important)
    """

    rewards = []

    for completion, ans in zip(completions, answer):
        comp = completion[0]["content"] if isinstance(completion, list) else completion

        # --- Individual rewards ---
        r_format  = reward_format(comp)
        r_cwe     = reward_cwe_match(comp, ans)
        r_keyword = reward_keyword_alignment(comp, ans)

        # --- Weighted combination ---
        total_reward = (
            0.5 * r_cwe +        # correctness (dominant signal)
            0.3 * r_format +     # structure + reasoning
            0.2 * r_keyword      # semantic grounding
        )

        wandb.log({
            "reward/format": r_format,
            "reward/cwe": r_cwe,
            "reward/keyword": r_keyword,
            "reward/total": total_reward
        })

        rewards.append(total_reward)

    return rewards