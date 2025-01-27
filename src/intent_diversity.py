"""Algorithms to order the page based on intent diversity of the paper
https://arxiv.org/pdf/2405.12327."""

from typing import List
import numpy as np


def order_page_intents(
    itemids: List[int],  # Shape [N]
    pwt: np.ndarray,  # Shape [V]
    vms: np.ndarray,  # Shape [N]
    es: np.ndarray,  # Shape [N, V]
    maxvals: np.ndarray,  # Shape [V]
    iwt: np.ndarray,  # Shape [V]
    pagelen: int,
    debug: bool = False,
) -> List[int]:
    """
    Order the page based on the maximum diversity of intents.

    Args:
    - itemids: List of item ids
    - pwt: VM weights for each intent
    - vms: Value model scores for each item
    - es: Event scores for each item and intent
    - maxvals: Maximum values for normalization
    - iwt: Normalized weights for each intent = pwt * maxvals
    - pagelen: Number of items to select per page

    Returns:
    - A list of item ids ordered by intent diversity

    Implementation:
        1. Assume cwt is set in the beginning to pwt * 1/maxvals to indicate
            the importance of each intent. pwt is the importance of the intent.
            maxvals is the maximum value of the estimated score of that intent.
            Hence pwt * 1/maxvals * es[i, v] will be scores that are roughly
            proportional to the importance of the intent.
        2. for each intent v, compute the headroom, sum of the top pagelen
            scores for that intent.
        3. While we have not filled the page, select the item with the highest
            vms[i] * Sum_v ( es[i, v] * cwt[v] ) and update the headroom for
            each intent. Using headroom update cwt.
    """
    # Normalize the intent weights to probabilities
    cwt = pwt / np.sum(pwt)

    # Since we are changing es and vms in the process, we will make a copy
    es_a = np.array(es)  # Shape [N, V]
    vms_a = np.array(vms)

    # For each intent, compute the sum of the maximum pagelen es values
    # of that intent.
    es_headroom = np.sum(
        np.sort(es_a, axis=0)[-pagelen:, :], axis=0
    )  # Shape [V]

    # Compute the vms headroom as well
    vm_headroom = np.sort(vms_a)[-pagelen:].sum()

    # Initialize the ordered itemids
    ordered_itemids = []

    # Iterate until the page is filled
    while len(ordered_itemids) < pagelen:
        # Pick the item with the highest vms_a[i] * sum_v(es_a[i, v] * cwt[v])
        # Compute the partial term sum_{v}(es_a[i, v] * cwt[v]) for each item
        # shape [N], since cwt is shape [V].
        intent_part = np.einsum("ij,j->i", es_a, cwt)  # shape [N]
        # Or: intent_part = np.sum(es_a * cwt, axis=1)

        # Compute the scores for each item
        current_scores = vms_a * intent_part  # shape [N]
        if debug:
            print(f"Scores being used to select: {current_scores}")

        # Select the item with the highest score
        next_best_item = np.argmax(current_scores)
        if debug:
            print(f"Next best item: {next_best_item}")
        # In debug case also print the item with maximum vm score
        if debug:
            print(f"Current vm scores: {vms_a}")
            print(f"Item with maximum vm score: {np.argmax(vms_a)}")

        # Add the selected item to the ordered itemids
        ordered_itemids.append(next_best_item)

        # 3) Update cwt using the formula:
        #    cwt[v] = cwt[v] * ( (es_headroom[v] - es_a[next_best_item, v]) / es_headroom[v] ) / (1 - vms_a[next_best_item]/vm_headroom)
        denom = (vm_headroom - vms_a[next_best_item]) / vm_headroom
        # Floor the denom to 1e-6 to avoid dividing by zero or negative
        denom = max(denom, 1e-6)

        numerator = (
            es_headroom - es_a[next_best_item, :]
        ) / es_headroom  # shape [V]
        cwt *= numerator / denom  # vectorized update across all v
        if debug:
            print(f"Updated cwt: {cwt}")

        # 4) Update headrooms
        es_headroom -= es_a[next_best_item, :]
        vm_headroom -= vms_a[next_best_item]

        # 5) Zero out that row in es_a and vms_a so it’s not chosen again
        es_a[next_best_item, :] = 0.0
        vms_a[next_best_item] = 0.0

    return ordered_itemids
