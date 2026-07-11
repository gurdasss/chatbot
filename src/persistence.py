import json

import numpy as np


def save_weights(
    hidden_weights: np.ndarray,
    output_weights: np.ndarray,
    filename: str = "model.json",
) -> None:
    # numpy arrays aren't JSON-serializable, so turn them into plain lists first
    data = {
        "hidden_weights": hidden_weights.tolist(),
        "output_weights": output_weights.tolist(),
    }

    with open(filename, "w") as f:
        json.dump(data, f)


def load_weights(filename: str = "model.json") -> tuple[np.ndarray, np.ndarray]:
    with open(filename) as f:
        data = json.load(f)

    # turn the plain lists back into numpy arrays for training/inference
    hidden_weights: np.ndarray = np.array(data["hidden_weights"])
    output_weights: np.ndarray = np.array(data["output_weights"])

    return (hidden_weights, output_weights)
