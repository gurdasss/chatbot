from random import uniform

examples = [
    (1, 1, 7),
    (2, 3, 15),
    (3, 1, 11),
    (5, 2, 19),
    (1, 4, 17),
]
LEARNING_RATE: float = 0.001
MAX_WEIGHT_LIMIT: float = 7.0
MIN_WEIGHT_LIMIT: float = -7.0


def cal_prediction(
    input_a: float, guess_a: float, input_b: float, guess_b: float
) -> float:
    return input_a * guess_a + input_b * guess_b


def cal_error(prediction: float, output: float) -> float:
    return prediction - output


def cal_guess(guess: float, error: float, scale: float) -> float:
    return guess - error * LEARNING_RATE * scale


w1: float = uniform(MIN_WEIGHT_LIMIT, MAX_WEIGHT_LIMIT)
w2: float = uniform(MIN_WEIGHT_LIMIT, MAX_WEIGHT_LIMIT)
w3: float = uniform(MIN_WEIGHT_LIMIT, MAX_WEIGHT_LIMIT)
w4: float = uniform(MIN_WEIGHT_LIMIT, MAX_WEIGHT_LIMIT)
w5: float = uniform(MIN_WEIGHT_LIMIT, MAX_WEIGHT_LIMIT)
w6: float = uniform(MIN_WEIGHT_LIMIT, MAX_WEIGHT_LIMIT)


for _ in range(100):
    for p in examples:
        input_a: float = p[0]
        input_b: float = p[1]
        output: float = p[-1]
        neuron_a_predict: float = cal_prediction(input_a, w1, input_b, w2)
        neuron_b_predict: float = cal_prediction(input_a, w3, input_b, w4)
        neuron_c_predict: float = cal_prediction(
            neuron_a_predict, w5, neuron_b_predict, w6
        )
        neuron_c_error: float = cal_error(neuron_c_predict, output)
        neuron_a_backprop_error = neuron_c_error * w5
        neuron_b_backprop_error = neuron_c_error * w6

        w1 = cal_guess(w1, neuron_a_backprop_error, input_a)
        w2 = cal_guess(w2, neuron_a_backprop_error, input_b)
        w3 = cal_guess(w3, neuron_b_backprop_error, input_a)
        w4 = cal_guess(w4, neuron_b_backprop_error, input_b)
        w5 = cal_guess(w5, neuron_c_error, neuron_a_predict)
        w6 = cal_guess(w6, neuron_c_error, neuron_b_predict)

        print(f"Expected: {output}, Got: {neuron_c_predict:.2f}")
