import random

# Each sequence is a "conversation"
# Think of each number as a Morse token
data = [
    [0, 1, 2, 3],
    [0, 1, 2, 3],
    [4, 5, 3, 1],
    [0, 1, 4, 5],
    [4, 5, 2, 3],
]

tally = {}

for seq in data:
    for i in range(len(seq) - 2):
        # Keeping the keys un-formatted
        # So that I can easily fetch tokens
        key: str = f"{seq[i]}{seq[i+1]}{seq[i+2]}"
        if key not in tally:
            tally[key] = 0

        tally[key] += 1


def predict(a: int, b: int) -> int:

    prediction_count_pair: list[tuple[int, int]] = []
    for key in tally.keys():
        first: int = int(key[0])
        second: int = int(key[1])

        # We need the exact two sequences
        if a == first and b == second:
            # Append the prediction and the count
            prediction_count_pair.append((int(key[-1]), tally[key]))

    final_prediction: int = None
    highest_count: int = -1
    for prediction, count in prediction_count_pair:
        if highest_count < count:
            highest_count = count
            final_prediction = prediction
        elif highest_count == count:  # it's a tie, pick randomly
            final_prediction = random.choice(
                list(map(lambda p: p[0], prediction_count_pair))
            )

    return final_prediction


# print(predict(0, 1))
# print(predict(1, 2))  # should give 3
# print(predict(4, 5))  # should be uncertain — 3 or 2
# print(predict(5, 3))  # should give 1


def generate() -> None:

    # Initial pair
    a: int = 0
    b: int = 1
    prediction: int = predict(a, b)

    for i in range(10):
        print(f"({a}, {b}) -> {prediction}")
        a = b
        b = prediction
        prediction = predict(a, b)


generate()
