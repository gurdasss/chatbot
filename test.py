import random

# Each sequence is a "conversation"
# Think of each number as a Morse token
data = [
    [0, 1, 2, 3],
    [0, 1, 2, 3],
    [4, 5, 3, 1],
    [0, 1, 4, 5],
    [4, 5, 2, 3],
    # New ones
    [4, 3, 0, 2],
    [3, 2, 0, 1],
    [3, 2, 5, 4],
    [5, 1, 3, 4],
    [2, 0, 5, 4],
]

trigram_tally = {}
bigram_tally = {}

# Build trigram model
for seq in data:
    for i in range(len(seq) - 2):
        # Keeping the keys un-formatted
        # So that I can easily fetch tokens
        key: str = f"{seq[i]}{seq[i+1]}{seq[i+2]}"
        if key not in trigram_tally:
            trigram_tally[key] = 0

        trigram_tally[key] += 1

# Build bigram model
for seq in data:
    for i in range(len(seq) - 1):
        key = f"{seq[i]} -> {seq[i+1]}"
        if key not in bigram_tally:
            bigram_tally[key] = 0
        bigram_tally[key] += 1


def trigram_predict(a: int, b: int) -> int:

    prediction_count_pair: list[tuple[int, int]] = []
    for key in trigram_tally.keys():
        first: int = int(key[0])
        second: int = int(key[1])

        # We need the exact two sequences
        if a == first and b == second:
            # Append the prediction and the count
            prediction_count_pair.append((int(key[-1]), trigram_tally[key]))

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


def bigram_predict(n: int) -> int:

    possible_keys: list[str] = []
    # Find all the keys with first == n
    for key in bigram_tally.keys():
        first: str = key[0]
        if str(n) == first:
            possible_keys.append(key)

    highest_count: int = -1
    prediction: int = None

    a = list(map(lambda key: (key[-1], bigram_tally[key]), possible_keys))

    for prediction_a, count in a:
        if highest_count < count:
            highest_count = count
            prediction = int(prediction_a)
        elif highest_count == count:
            prediction = random.choice(list(map(lambda pair: int(pair[0]), a)))

    return prediction


def generate() -> None:

    # Initial pair
    a: int = 0
    b: int = 1
    prediction: int = trigram_predict(a, b)

    for i in range(10):

        # if we run out of trigram prediction
        # then we'll use the last element and
        # predict using bigram_predict
        if prediction == None:
            prediction = bigram_predict(b)

        print(f"({a}, {b}) -> {prediction}")
        a = b
        b = prediction
        prediction = trigram_predict(a, b)


generate()
print(f"\n{trigram_tally}")
