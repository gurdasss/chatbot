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

bigram_tally: dict = {}

# First pair and count things up
for seq in data:
    for i in range(len(seq) - 1):
        key: tuple[int, int] = (seq[i], seq[i + 1])
        if key not in bigram_tally:
            bigram_tally[key] = 0
        bigram_tally[key] += 1

# Second, build the number line for each key
key_no_line = {}

for item in bigram_tally.items():
    key: int = item[0][0]
    prediction: int = item[0][-1]
    current_count: int = item[-1]

    # First create a key with an initial pair as its value
    if key not in key_no_line:
        key_no_line[key] = [(prediction, current_count)]
    else:
        cumulative_count: int = key_no_line[key][-1][-1]
        key_no_line[key] += [(prediction, current_count + cumulative_count)]

    # print(f"{key}, {prediction}, {current_count}")

print(key_no_line)


def bigram_sample(n: int) -> int:
    if n not in key_no_line:
        return -1

    cumulative_count: int = key_no_line[n][-1][-1]
    # Random sample between 0 to cumulative_count (excluding)
    random_sample: float = random.uniform(0, cumulative_count)
    final_prediction: int = None
    # print(random_sample)

    for p in key_no_line[n]:
        prediction: int = p[0]
        count: float = p[-1]

        # find the first prediction whose cumulative_count
        # is below the random sample we got
        if random_sample < count:
            final_prediction = prediction
            break

    return final_prediction


def bigram_predict(n: int) -> int:

    prediction_count_pair: list[tuple[int, int]] = []
    for key in bigram_tally.keys():
        if n == int(key[0]):  # Look at the very first token
            # Append the prediction and the count
            prediction_count_pair.append((int(key[-1]), bigram_tally[key]))

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


# print(bigram_predict(0))  # 1
# print(bigram_predict(1))  # 2
# print(bigram_predict(5))  # not sure, 2 or 3
# print(bigram_predict(5))  # not sure, 2 or 3
# print(bigram_predict(5))  # not sure, 2 or 3
# print(bigram_predict(5))  # not sure, 2 or 3
# print(bigram_predict(5))  # not sure, 2 or 3
# print(bigram_predict(5))  # not sure, 2 or 3
# print(bigram_predict(5))  # not sure, 2 or 3
# print(bigram_predict(5))  # not sure, 2 or 3
# print(bigram_predict(5))  # not sure, 2 or 3
# print(bigram_predict(5))  # not sure, 2 or 3
# print(bigram_predict(5))  # not sure, 2 or 3
# print(bigram_predict(5))  # not sure, 2 or 3
# print(bigram_predict(5))  # not sure, 2 or 3
# print(bigram_predict(5))  # not sure, 2 or 3
