from math import exp

from src.codec import encode, idx_to_char
from src.dataset import make_pairs
from src.network import (
    CONTEXT_WINDOW,
    NO_OF_HIDDEN_NEURONS,
    NO_OF_TOKENS,
    forward,
    init_weights,
    predict,
)

LEARNING_RATE: float = 0.001


def train_one_example(
    encoded_inputs: list[int],
    expected_output: int,
    hidden_weights: list[list[float]],
    output_weights: list[list[float]],
) -> None:
    hidden_values, output_scores = forward(
        encoded_inputs, hidden_weights, output_weights
    )

    # Right now the output scores can be negative, zero, or positive
    # Probabilities can't be negative and so, we need to convert it to
    # positive first with softmax method

    max_score = max(output_scores)
    scores_raise_by_e: list[float] = [exp(score - max_score) for score in output_scores]

    # Step 1: `e`(2.718) to the power of each score
    # scores_raise_by_e = list(map(lambda score: 2.718**score, output_scores))
    # Step 2: accumulate into final sum
    final_sum: float = sum(scores_raise_by_e)
    # Step 3: divide the list by the final sum
    scores_raise_by_e = list(map(lambda score: score / final_sum, scores_raise_by_e))

    # Build the target list with all zeros except the index at expected output
    targets: list[int] = [0] * NO_OF_TOKENS
    targets[expected_output] = 1

    # Calculate the errors
    errors: list[float] = [
        probability - target for probability, target in zip(scores_raise_by_e, targets)
    ]

    for i in range(NO_OF_TOKENS):
        for j in range(NO_OF_HIDDEN_NEURONS):
            output_weights[i][j] = (
                output_weights[i][j] - errors[i] * LEARNING_RATE * hidden_values[j]
            )

    # Step 7: backpropagate errors to hidden layer
    hidden_errors = []
    for j in range(NO_OF_HIDDEN_NEURONS):
        error_sum = 0
        for i in range(NO_OF_TOKENS):
            error_sum += errors[i] * output_weights[i][j]
        hidden_errors.append(error_sum)

    # Step 8: adjust hidden weights
    for j in range(NO_OF_HIDDEN_NEURONS):
        for k in range(CONTEXT_WINDOW):
            hidden_weights[j][k] = (
                hidden_weights[j][k]
                - hidden_errors[j] * LEARNING_RATE * encoded_inputs[k]
            )


def generate(seed_text, hidden_weights, output_weights, length=20):
    current = encode(seed_text[:CONTEXT_WINDOW])
    result = seed_text[:CONTEXT_WINDOW]

    for _ in range(length):
        hidden_values, output_scores = forward(current, hidden_weights, output_weights)
        next_token = predict(output_scores)
        result += idx_to_char[next_token]
        current = current[1:] + [next_token]

    return result


def main() -> None:
    hidden_weights, output_weights = init_weights()
    text = "how are you doing today i am doing well thank you"
    pairs = make_pairs(text)

    for _ in range(5000):
        for input_tokens, expected in pairs:
            train_one_example(input_tokens, expected, hidden_weights, output_weights)

    # Test every pair
    for input_tokens, expected in pairs:
        _, scores = forward(input_tokens, hidden_weights, output_weights)
        predicted = idx_to_char[scores.index(max(scores))]
        expected_char = idx_to_char[expected]
        print(f"Expected: {expected_char}, Got: {predicted}")

    print(generate("how are ", hidden_weights, output_weights))


if __name__ == "__main__":
    main()
