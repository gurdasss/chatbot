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

bigram_tally = {}

for seq in data:
    for i in range(len(seq) - 1):
        key: tuple[int, int] = (seq[i], seq[i + 1])
        if key not in bigram_tally:
            bigram_tally[key] = 0
        bigram_tally[key] += 1


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


print(bigram_predict(0))  # 1
print(bigram_predict(1))  # 2
print(bigram_predict(5))  # not sure, 2 or 3
print(bigram_predict(5))  # not sure, 2 or 3
print(bigram_predict(5))  # not sure, 2 or 3
print(bigram_predict(5))  # not sure, 2 or 3
print(bigram_predict(5))  # not sure, 2 or 3
print(bigram_predict(5))  # not sure, 2 or 3
print(bigram_predict(5))  # not sure, 2 or 3
print(bigram_predict(5))  # not sure, 2 or 3
print(bigram_predict(5))  # not sure, 2 or 3
print(bigram_predict(5))  # not sure, 2 or 3
print(bigram_predict(5))  # not sure, 2 or 3
print(bigram_predict(5))  # not sure, 2 or 3
print(bigram_predict(5))  # not sure, 2 or 3
print(bigram_predict(5))  # not sure, 2 or 3
