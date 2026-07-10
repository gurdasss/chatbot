examples: list[tuple] = [(2, 6), (5, 15), (10, 30)]
guess: float = 7.0
LEARNING_RATE: float = 0.1

for _ in range(100):
    for p in examples:
        input_val: float = p[0]
        output_val: float = p[-1]
        prediction: float = input_val * guess
        error: float = prediction - output_val
        # For each errors, adjust the guess based
        # on the learning rate. If positive, reduce a
        # little and vice versa.
        guess = guess - error * LEARNING_RATE

print(guess)
