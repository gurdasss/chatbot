import os

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
from src.persistence import load_weights, save_weights

LEARNING_RATE: float = 0.00001
MODEL_FILE: str = "model.json"
TRAINING_DATA_FILE: str = "training_data.txt"


def train_one_example(
    encoded_inputs: list[int],
    expected_output: int,
    hidden_weights: np.ndarray,
    output_weights: np.ndarray,
) -> bool:
    hidden_values, output_scores = forward(
        encoded_inputs, hidden_weights, output_weights
    )

    # Catch blown-up weights early instead of finding out at the end
    if np.isnan(output_scores).any():
        print("NaN detected, stopping training")
        return True

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
    # Cap the error signal so one example can't blow up the weights
    errors = np.clip(errors, -1.0, 1.0)

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

    return False


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
    # all training text lives in this file, one line per example. The
    # vocabulary has no newline token, so join lines with a space instead.
    with open(TRAINING_DATA_FILE) as f:
        text = " ".join(line.strip() for line in f if line.strip())
    pairs = make_pairs(text)

    if os.path.exists(MODEL_FILE):
        # reuse a previously trained model instead of starting from scratch
        print(f"Loading weights from {MODEL_FILE}")
        hidden_weights, output_weights = load_weights(MODEL_FILE)
    else:
        hidden_weights, output_weights = init_weights()

        for _ in range(2000):
            stop_training = False
            for input_tokens, expected in pairs:
                stop_training = train_one_example(
                    input_tokens, expected, hidden_weights, output_weights
                )
                if stop_training:
                    break
            if stop_training:
                break

        # persist the trained weights so future runs can skip training
        save_weights(hidden_weights, output_weights, MODEL_FILE)
        print(f"Saved weights to {MODEL_FILE}")

    # Test every pair
    for input_tokens, expected in pairs:
        _, scores = forward(input_tokens, hidden_weights, output_weights)
        # pick the token with the highest score
        predicted = idx_to_char[int(scores.argmax())]
        expected_char = idx_to_char[expected]
        # print(f"Expected: {expected_char}, Got: {predicted}")

    print(generate("hello wo", hidden_weights, output_weights))


if __name__ == "__main__":
    main()
