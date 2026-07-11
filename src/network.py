from random import random

import numpy as np

MAX_WEIGHT: float = 0.5
MIN_WEIGHT: float = -0.5

CONTEXT_WINDOW: int = 8
NO_OF_HIDDEN_NEURONS: int = 64
NO_OF_TOKENS: int = 28


def init_weights() -> tuple[np.ndarray, np.ndarray]:
    # Stack one weight row per neuron into a single grid of numbers.
    hidden_weights: np.ndarray = np.array(
        [gen_weights(CONTEXT_WINDOW) for _ in range(NO_OF_HIDDEN_NEURONS)]
    )
    output_weights: np.ndarray = np.array(
        [gen_weights(NO_OF_HIDDEN_NEURONS) for _ in range(NO_OF_TOKENS)]
    )

    return (hidden_weights, output_weights)


def gen_weights(n: int) -> np.ndarray:
    # A row of n random starting weights.
    return np.random.uniform(MIN_WEIGHT, MAX_WEIGHT, size=n)


def forward(
    encoded_inputs: list[int],
    hidden_weights: np.ndarray,
    output_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inputs: np.ndarray = np.array(encoded_inputs, dtype=float)

    # Step 1: compute 64 hidden values in one shot.
    # This is the same as multiplying each neuron's weights by the
    # inputs and adding them up, just done for every neuron at once.
    hidden_values: np.ndarray = hidden_weights @ inputs
    # That's called ReLU. It prevents negative values from
    # cascading through the network and destabilizing everything.
    hidden_values = np.maximum(0.0, hidden_values)

    # Step 2: compute 28 output scores, again all at once.
    output_scores: np.ndarray = output_weights @ hidden_values
    # Return both (you'll need hidden values later for training)

    return (hidden_values, output_scores)


def predict(output_scores: np.ndarray) -> int:

    # Right now the output scores can be negative, zero, or positive
    # Probabilities can't be negative and so, we need to convert it to
    # positive first with softmax method

    output_scores = np.asarray(output_scores, dtype=float)
    max_score = np.max(output_scores)
    # Step 1: `e`(2.718) to the power of each score, for every score at once
    scores_raise_by_e: np.ndarray = np.exp(output_scores - max_score)

    # Step 2: accumulate into final sum
    final_sum: float = np.sum(scores_raise_by_e)
    # Step 3: divide the list by the final sum
    scores_raise_by_e = scores_raise_by_e / final_sum

    # random sample between [0, 1)
    random_sample: float = random()
    prediction: int = None
    cumulative_probability: float = scores_raise_by_e[0]

    for i in range(len(scores_raise_by_e) - 1):

        # find the first prediction who
        # is above the random sample we got
        if cumulative_probability > random_sample:
            prediction = i
            break

        cumulative_probability += scores_raise_by_e[i + 1]

    if prediction is None:
        prediction = len(scores_raise_by_e) - 1

    return prediction
