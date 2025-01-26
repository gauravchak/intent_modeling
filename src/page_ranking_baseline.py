"""Algorithms to order the page based on the ranking algorithm."""

from typing import List
import numpy as np


def order_page(
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
    Order the page based on the ranking algorithm.
    We are not implementing the actual ranking algorithm here.
    We are just returning the top pagelen items.
    """
    # A placeholder implementation
    return itemids[:pagelen]


def order_page_vm(
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
    Order the page based on the maximum vms scores.
    vms is the combined score for each candidate. This is akin to `s_ij` in
    equation 5 of the paper https://arxiv.org/pdf/2405.12327.
    """
    # Combine itemids and vms into a list of tuples
    item_vms = list(zip(itemids, vms))

    # Sort the list in descending order based on vms
    sorted_item_vms = sorted(item_vms, key=lambda x: x[1], reverse=True)

    # Return the top pagelen itemids
    return [item[0] for item in sorted_item_vms[:pagelen]]
