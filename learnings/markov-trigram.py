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

trigram_tally = {}

# First pair and count things up
for seq in data:
    for i in range(len(seq) - 2):
        key: tuple[int, int, int] = (seq[i], seq[i + 1], seq[i + 2])
        if key not in trigram_tally:
            trigram_tally[key] = 0
        trigram_tally[key] += 1

print(trigram_tally)

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

print(trigram_key_no_line)


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


print(trigram_predict(0, 1))
print(trigram_predict(1, 2))  # should give 3
print(trigram_predict(4, 5))  # should be uncertain — 3 or 2
print(trigram_predict(5, 3))  # should give 1


def generate() -> None:

    # Initial pair
    a: int = 0
    b: int = 1
    prediction: int = trigram_predict(a, b)

    for i in range(10):
        print(f"({a}, {b}) -> {prediction}")
        a = b
        b = prediction
        prediction = trigram_predict(a, b)


generate()
