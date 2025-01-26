from typing import List
import numpy as np

from gen_data import generate_data
from page_ranking_baseline import order_page, order_page_vm
from intent_diversity import order_page_intents
from eval import evaluate_ranking


def run_evals(
    num_evals: int,
    num_intents: int,
    num_candidates: int,
    pagelen: int,
    debug: bool = False,
) -> np.ndarray:
    order_page_funcs = [
        # order_page,
        # order_page_vm,
        order_page_intents
    ]
    scores_array = np.zeros((num_evals, len(order_page_funcs)), dtype=float)
    for i in range(num_evals):
        # Generate data
        (es, maxvals, pwt, vms, iwt) = generate_data(
            num_candidates=num_candidates, num_intents=num_intents, debug=debug
        )
        if debug:
            print(f"Event scores (es): {es}")
            print(f"Max values: {maxvals}")
            print(f"VM weights (pwt): {pwt}")
            print(f"Value model scores (vms): {vms}")
            print(f"Intent weights (iwt): {iwt}")

        # Define parameters
        itemids = list(range(num_candidates))

        # Evaluate ranking
        # order_page, order_page_vm, order_page_intents
        for j, f in enumerate(order_page_funcs):
            score = evaluate_ranking(
                itemids=itemids,
                pwt=pwt,
                vms=vms,
                es=es,
                maxvals=maxvals,
                iwt=iwt,
                pagelen=pagelen,
                order_page_func=f,
                debug=debug,
            )
            if debug:
                print(f"Score ({f.__name__}):", score)
            scores_array[i, j] = score

    # Compute the means
    mean_scores = np.mean(scores_array, axis=0)
    if debug:
        print(f"Scores: {scores_array}")
        print(f"Mean scores: {mean_scores}")
    return mean_scores


mean_scores = run_evals(
    num_evals=1,
    num_intents=3,
    num_candidates=10,
    pagelen=2,
    debug=True,
)
# Divide all scores by the first score (baseline)
increase_from_baseline = mean_scores / mean_scores[0]

# Round the arrays to 3 decimal places
rounded_mean_scores = np.round(mean_scores, 3)
rounded_increase_from_baseline = np.round(increase_from_baseline, 3) - 1

# Print the formatted arrays
print(f"Scores: {rounded_mean_scores}")
print(f"Increase from baseline: {rounded_increase_from_baseline}")
