from random import uniform

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
