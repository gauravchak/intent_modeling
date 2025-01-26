"""generate data for evaluation"""

from typing import Tuple
import numpy as np


def generate_data(
    num_candidates: int,
    num_intents: int,
    intent_wt_random: bool = True,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data for the problem.

    Parameters:
    num_candidates (int): Number of input candidates
    num_intents (int): Number of intents / scores per candidate
    debug (bool): Whether to print debug information

    Returns:
    es (np.ndarray): Event scores of shape (N, V)
    maxvals (np.ndarray): List of maximum values for each score
    pwt (np.ndarray): Weights of shape (V,). These can be thought of value
        model weights. Except these are not normalized. To actually use them
        as weights, you need to multiply then with say 1/maxvals[v] so that
        you can then use them as weights in the weighted sum of es.
    vms (np.ndarray): value model scores (N,). Combined score for each
        candidate.
    intent_weight (np.ndarray): Weights for each intent. The final VM score[i]
        is the sum of es[i, v] * intent_weight[v] / maxval[v] for all v. This
        is because pwt[v] = intent_weight[v] / maxval[v] and vms[i] = sum of
        es[i, v] * pwt[v] for all v.

    Example usage:
    es, maxvals, pwt, vms, iwt = generate_data(
        num_candidates=100, num_intents=5, debug=False
    )
    """
    # Step 1: Generate event scores "es" with shape N=100, V=5
    # Calculate alpha such that alpha^10 = 0.5
    alpha = np.power(0.5, 1 / 10)

    es = np.zeros((num_candidates, num_intents))
    maxvals = np.empty(num_intents)  # Shape [V]

    for v in range(num_intents):
        # Choose a random b between 1 and 3.5
        b = np.random.uniform(1, 3.5)
        maxval = np.power(10, -b)
        maxvals[v] = maxval

        # Generate and shuffle es[i, v] = maxval * alpha^i for each i
        powers = maxval * np.power(alpha, np.arange(num_candidates))

        # Shuffle the powers array to make ranking nontrivial
        np.random.shuffle(powers)

        es[:, v] = powers

    # Step 2: Generate pwt
    # Part 1 is to generate intent weights of shape V
    if intent_wt_random:
        # either with values randomly from 0.3 to 3
        intent_weight = np.random.uniform(0.3, 3, num_intents)
    else:
        # or with equal weights
        intent_weight = np.ones(num_intents)

    # Compute pwt by dividing intent_weight by maxvals used to generate the es.
    # This ensures that the importance of each intent is proportional to
    # intent_weight. This is because pwt * the maximum es value for that intent
    # will then be proportional to intent_weight.
    pwt = intent_weight / maxvals

    # Step 3: Combine es[N, V] with weights pwt[V] to make vms[N]
    vms = np.dot(es, pwt)
    return es, maxvals, pwt, vms, intent_weight
