guess = 7.0

examples = [(2, 6), (5, 15), (10, 30), (20, 60)]

for i in range(100):
    for input_val, expected in examples:
        prediction = guess * input_val
        error = prediction - expected
        guess = guess - error * 0.001

print(guess)
