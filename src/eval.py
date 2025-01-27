from typing import List
import numpy as np


def evaluate_ranking(
    itemids: List[int],
    pwt: np.ndarray,
    vms: np.ndarray,
    es: np.ndarray,
    maxvals: np.ndarray,
    iwt: np.ndarray,
    pagelen: int,
    order_page_func: callable,
    debug: bool = False,
) -> float:
    """
    Evaluate the performance of the ranking algorithm.

    Parameters:
    - itemids: List of item ids
    - pwt: VM weights for each intent
    - vms: Value model scores for each item
    - es: Event scores for each item and intent
    - maxvals: Maximum values for normalization
    - iwt: Normalized weights for each intent = pwt * maxvals.
    - pagelen: Number of items to select per page

    Returns:
    - A float representing the evaluation score
    """
    # Normalize the intent weights to probabilities
    intent_probs = np.array(pwt) / sum(pwt)

    # Select an intent with probability proportional to its weight
    selected_intent = np.random.choice(len(pwt), p=intent_probs)

    if debug:
        print(f"Selected intent: {selected_intent}")

    # Order the page
    ordered_itemids = order_page_func(
        itemids=itemids,
        pwt=pwt,
        vms=vms,
        es=es,
        maxvals=maxvals,
        iwt=iwt,
        pagelen=pagelen,
        debug=debug,
    )

    if debug:
        print(f"Ordered itemids: {ordered_itemids}")

    # Calculate the sum of event scores for the selected items
    numerator = sum(es[itemid][selected_intent] for itemid in ordered_itemids)

    # Calculate the sum of the top pagelen event scores for the selected intent
    sorted_scores = sorted(es[i][selected_intent] for i in itemids)[::-1][
        :pagelen
    ]
    denominator = sum(sorted_scores)

    # Evaluation metric
    score = numerator / denominator

    return score
