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
