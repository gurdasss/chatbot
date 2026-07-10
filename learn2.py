examples = [
    (1, 1, 6),  # 1*2 + 1*4 = 6
    (2, 3, 16),  # 2*2 + 3*4 = 16
    (3, 1, 10),  # 3*2 + 1*4 = 10
    (5, 2, 18),  # 5*2 + 2*4 = 18
    (1, 4, 18),  # 1*2 + 4*4 = 18
]

guess_a: float = 7.0
guess_b: float = 7.0
LEARNING_RATE: float = 0.1

for _ in range(100):
    for p in examples:
        input_a: float = p[0]
        input_b: float = p[1]
        output: float = p[-1]
        prediction: float = input_a * guess_a + input_b * guess_b
        error: float = prediction - output

        # Adjust the both the guesses based upon their respective input's scale
        guess_a = guess_a - error * LEARNING_RATE * input_a
        guess_b = guess_b - error * LEARNING_RATE * input_b

        # I needed to adjust both of the guesses based upon
        # by how much amount each guesses has been multiplied
        # by their respective input
        # print(f"Previous error: {error}")
        # # Test 1: Let's adjust guess_a and see, if the error reduces!
        # guess_a = guess_a - error * LEARNING_RATE
        # prediction = input_a * guess_a + input_b * guess_b
        # error = prediction - output
        # print(f"Now error: {error}")


print(guess_a, guess_b)
