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

# for seq in data:
#     for i in range(len(seq) - 1):
#         tally[f"{seq[i]} -> {seq[i+1]}"] = 0

# for seq in data:
#     for i in range(len(seq) - 1):
#         tally[f"{seq[i]} -> {seq[i+1]}"] += 1

for seq in data:
    for i in range(len(seq) - 1):
        key = f"{seq[i]} -> {seq[i+1]}"
        if key not in tally:
            tally[key] = 0
        tally[key] += 1

# print(tally)

import random


def predict(n: int) -> int:

    possible_keys: list[str] = []
    # Find all the keys with first == n
    for key in tally.keys():
        first: str = key[0]
        if str(n) == first:
            possible_keys.append(key)

    highest_count: int = -1
    prediction: int = None

    a = list(map(lambda key: (key[-1], tally[key]), possible_keys))

    for prediction_a, count in a:
        if highest_count < count:
            highest_count = count
            prediction = int(prediction_a)
        elif highest_count == count:
            prediction = random.choice(list(map(lambda pair: int(pair[0]), a)))

    return prediction


print(predict(0))  # 1
print(predict(1))  # 2
print(predict(5))  # not sure, 2 or 3
print(predict(5))  # not sure, 2 or 3
print(predict(5))  # not sure, 2 or 3
print(predict(5))  # not sure, 2 or 3
print(predict(5))  # not sure, 2 or 3
print(predict(5))  # not sure, 2 or 3
print(predict(5))  # not sure, 2 or 3
print(predict(5))  # not sure, 2 or 3
print(predict(5))  # not sure, 2 or 3
print(predict(5))  # not sure, 2 or 3
print(predict(5))  # not sure, 2 or 3
print(predict(5))  # not sure, 2 or 3
print(predict(5))  # not sure, 2 or 3
print(predict(5))  # not sure, 2 or 3
