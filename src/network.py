from random import uniform, random
from math import exp

MAX_WEIGHT: float = 0.5
MIN_WEIGHT: float = -0.5

CONTEXT_WINDOW: int = 8
NO_OF_HIDDEN_NEURONS: int = 64
NO_OF_TOKENS: int = 28


def init_weights() -> tuple[list[list[float]], list[list[float]]]:
    hidden_weights: list[list[float]] = []
    output_weights: list[list[float]] = []

    for _ in range(NO_OF_HIDDEN_NEURONS):
        hidden_weights.append(gen_weights(CONTEXT_WINDOW))

    for _ in range(NO_OF_TOKENS):
        output_weights.append(gen_weights(NO_OF_HIDDEN_NEURONS))

    return (hidden_weights, output_weights)


def gen_weights(n: int) -> list[float]:
    return [uniform(MIN_WEIGHT, MAX_WEIGHT) for _ in range(n)]


def forward(
    encoded_inputs: list[int],
    hidden_weights: list[list[float]],
    output_weights: list[list[float]],
) -> tuple[list[float], list[list[float]]]:
    # Step 1: compute 64 hidden values
    hidden_values: list[float] = []
    for neuron_output_scores in hidden_weights:
        neuron_output: float = sum(
            [
                encoded_input * neuron_weight
                for encoded_input, neuron_weight in zip(
                    encoded_inputs, neuron_output_scores
                )
            ]
        )
        hidden_values.append(neuron_output)

    output_scores: list[float] = []
    # Step 2: compute 28 output scores
    for neuron_output_scores in output_weights:
        neuron_output: float = sum(
            [
                hidden_value * neuron_weight
                for hidden_value, neuron_weight in zip(
                    hidden_values, neuron_output_scores
                )
            ]
        )
        output_scores.append(neuron_output)
    # Return both (you'll need hidden values later for training)

    return (hidden_values, output_scores)


def predict(output_scores: list[float]) -> int:

    # Right now the output scores can be negative, zero, or positive
    # Probabilities can't be negative and so, we need to convert it to
    # positive first with softmax method

    max_score = max(output_scores)
    # Step 1: `e`(2.718) to the power of each score
    scores_raise_by_e: list[float] = [exp(score - max_score) for score in output_scores]

    # scores_raise_by_e = list(map(lambda score: 2.718**score, output_scores))
    # Step 2: accumulate into final sum
    final_sum: float = sum(scores_raise_by_e)
    # Step 3: divide the list by the final sum
    scores_raise_by_e = list(map(lambda score: score / final_sum, scores_raise_by_e))

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
