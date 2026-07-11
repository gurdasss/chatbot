import numpy as np

from src.codec import encode, idx_to_char
from src.dataset import make_pairs
from src.network import (
    CONTEXT_WINDOW,
    NO_OF_TOKENS,
    forward,
    init_weights,
    predict,
)

LEARNING_RATE: float = 0.0001


def train_one_example(
    encoded_inputs: list[int],
    expected_output: int,
    hidden_weights: np.ndarray,
    output_weights: np.ndarray,
) -> None:
    hidden_values, output_scores = forward(
        encoded_inputs, hidden_weights, output_weights
    )

    # Right now the output scores can be negative, zero, or positive
    # Probabilities can't be negative and so, we need to convert it to
    # positive first with softmax method

    # Step 1: `e`(2.718) to the power of every score at once
    max_score = np.max(output_scores)
    probabilities: np.ndarray = np.exp(output_scores - max_score)
    # Step 2: accumulate into final sum
    final_sum: float = np.sum(probabilities)
    # Step 3: divide every score by the final sum
    probabilities = probabilities / final_sum

    # Build the target grid with all zeros except a one at the expected output
    targets: np.ndarray = np.zeros(NO_OF_TOKENS)
    targets[expected_output] = 1

    # Calculate the errors for every output at once
    errors: np.ndarray = probabilities - targets

    # Step 7: adjust output weights.
    # Every (token, hidden neuron) weight is nudged in one grid
    # operation instead of a pair of nested loops.
    output_weights -= LEARNING_RATE * np.outer(errors, hidden_values)

    # Step 8: backpropagate errors to hidden layer, all neurons at once
    hidden_errors: np.ndarray = errors @ output_weights
    # if a hidden neuron's output was clipped to zero,
    # its error should also be zero since it didn't
    # contribute anything.
    hidden_errors = np.where(hidden_values == 0, 0.0, hidden_errors)

    # Step 9: adjust hidden weights, same grid trick as step 7
    hidden_weights -= LEARNING_RATE * np.outer(hidden_errors, encoded_inputs)


def generate(seed_text, hidden_weights, output_weights, length=20):
    current = encode(seed_text[:CONTEXT_WINDOW])
    result = seed_text[:CONTEXT_WINDOW]

    for _ in range(length):
        hidden_values, output_scores = forward(current, hidden_weights, output_weights)
        next_token = predict(output_scores)
        result += idx_to_char[next_token]
        # slide the context window forward by one token
        current = current[1:] + [next_token]

    return result


def main() -> None:
    hidden_weights, output_weights = init_weights()
    text = "hello how are you-i am doing well thank you-how is your day going-my day is going great-what do you like to do-i like to read and code-that sounds like fun-it is really fun indeed"
    pairs = make_pairs(text)

    for _ in range(2000):
        for input_tokens, expected in pairs:
            train_one_example(input_tokens, expected, hidden_weights, output_weights)

    # Test every pair
    for input_tokens, expected in pairs:
        _, scores = forward(input_tokens, hidden_weights, output_weights)
        # pick the token with the highest score
        predicted = idx_to_char[int(scores.argmax())]
        expected_char = idx_to_char[expected]
        print(f"Expected: {expected_char}, Got: {predicted}")

    print(generate("i am doi", hidden_weights, output_weights))


if __name__ == "__main__":
    main()
