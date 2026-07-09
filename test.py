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

# First pair and count things up
for seq in data:
    for i in range(len(seq) - 2):
        key: tuple[int, int, int] = (seq[i], seq[i + 1], seq[i + 2])
        if key not in trigram_tally:
            trigram_tally[key] = 0
        trigram_tally[key] += 1


# Second, build the number line for each key
trigram_key_no_line = {}

for item in trigram_tally.items():
    key: tuple[int, int] = (item[0][0], item[0][1])
    prediction: int = item[0][-1]
    current_count: int = item[-1]

    # First create a key with an initial pair as its value
    if key not in trigram_key_no_line:
        trigram_key_no_line[key] = [(prediction, current_count)]
    else:
        cumulative_count: int = trigram_key_no_line[key][-1][-1]
        trigram_key_no_line[key] += [(prediction, current_count + cumulative_count)]


def trigram_sample(a: int, b: int) -> int:
    key: tuple[int, int] = (a, b)
    if key not in trigram_key_no_line:
        return -1

    cumulative_count: int = trigram_key_no_line[key][-1][-1]
    # Random sample between 0 to cumulative_count (excluding)
    random_sample: float = random.uniform(0, cumulative_count)
    final_prediction: int = None

    for p in trigram_key_no_line[key]:
        prediction: int = p[0]
        count: float = p[-1]

        # find the first prediction whose cumulative_count
        # is above the random sample we got
        if random_sample < count:
            final_prediction = prediction
            break

    return final_prediction


def trigram_predict(a: int, b: int) -> int:
    return trigram_sample(a, b)


bigram_tally = {}

# First pair and count things up
for seq in data:
    for i in range(len(seq) - 1):
        key: tuple[int, int] = (seq[i], seq[i + 1])
        if key not in bigram_tally:
            bigram_tally[key] = 0
        bigram_tally[key] += 1

# Second, build the number line for each key
bigram_key_no_line = {}

for item in bigram_tally.items():
    key: int = item[0][0]
    prediction: int = item[0][-1]
    current_count: int = item[-1]

    # First create a key with an initial pair as its value
    if key not in bigram_key_no_line:
        bigram_key_no_line[key] = [(prediction, current_count)]
    else:
        cumulative_count: int = bigram_key_no_line[key][-1][-1]
        bigram_key_no_line[key] += [(prediction, current_count + cumulative_count)]


def bigram_sample(n: int) -> int:
    if n not in bigram_key_no_line:
        return -1

    cumulative_count: int = bigram_key_no_line[n][-1][-1]
    # Random sample between 0 to cumulative_count (excluding)
    random_sample: float = random.uniform(0, cumulative_count)
    final_prediction: int = None

    for p in bigram_key_no_line[n]:
        prediction: int = p[0]
        count: float = p[-1]

        # find the first prediction whose cumulative_count
        # is above the random sample we got
        if random_sample < count:
            final_prediction = prediction
            break

    return final_prediction


def bigram_predict(n: int) -> int:
    return bigram_sample(n)


def generate() -> None:

    # Initial pair
    a: int = 0
    b: int = 1
    prediction: int = trigram_predict(a, b)

    for _ in range(10):

        # if we run out of trigram prediction
        # then we'll use the last element and
        # predict using bigram_predict
        if prediction == -1:
            prediction = bigram_predict(b)

        print(f"({a}, {b}) -> {prediction}")
        a = b
        b = prediction
        prediction = trigram_predict(a, b)


generate()
