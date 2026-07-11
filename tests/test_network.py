from unittest.mock import patch

import numpy as np

from src.network import (
    CONTEXT_WINDOW,
    MAX_WEIGHT,
    MIN_WEIGHT,
    NO_OF_HIDDEN_NEURONS,
    NO_OF_TOKENS,
    forward,
    gen_weights,
    init_weights,
    predict,
)


def test_init_weights_returns_two_lists():
    result = init_weights()
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_hidden_weights_shape():
    hidden_weights, _ = init_weights()
    assert isinstance(hidden_weights, np.ndarray)
    assert hidden_weights.shape == (NO_OF_HIDDEN_NEURONS, CONTEXT_WINDOW)
    assert hidden_weights.shape == (64, 8)


def test_output_weights_shape():
    _, output_weights = init_weights()
    assert isinstance(output_weights, np.ndarray)
    assert output_weights.shape == (NO_OF_TOKENS, NO_OF_HIDDEN_NEURONS)
    assert output_weights.shape == (28, 64)


def test_all_hidden_weights_within_bounds():
    hidden_weights, _ = init_weights()
    for neuron_weights in hidden_weights:
        for weight in neuron_weights:
            assert MIN_WEIGHT <= weight <= MAX_WEIGHT


def test_all_output_weights_within_bounds():
    _, output_weights = init_weights()
    for neuron_weights in output_weights:
        for weight in neuron_weights:
            assert MIN_WEIGHT <= weight <= MAX_WEIGHT


def test_weights_are_floats():
    hidden_weights, output_weights = init_weights()
    assert all(isinstance(w, float) for row in hidden_weights for w in row)
    assert all(isinstance(w, float) for row in output_weights for w in row)


def test_weights_are_randomized_not_constant():
    hidden_weights, output_weights = init_weights()
    # Extremely unlikely for random init to produce identical values throughout
    assert len(set(w for row in hidden_weights for w in row)) > 1
    assert len(set(w for row in output_weights for w in row)) > 1


def test_successive_calls_produce_different_weights():
    first_hidden, first_output = init_weights()
    second_hidden, second_output = init_weights()
    assert not np.array_equal(first_hidden, second_hidden)
    assert not np.array_equal(first_output, second_output)


def test_gen_weights_returns_requested_count_within_bounds():
    weights = gen_weights(10)
    assert len(weights) == 10
    assert all(MIN_WEIGHT <= w <= MAX_WEIGHT for w in weights)


def test_gen_weights_zero_length():
    assert len(gen_weights(0)) == 0


def test_forward_output_shapes_with_real_weights():
    inputs = [1, 2, 3, 4, 5, 6, 7, 8]
    hidden_weights, output_weights = init_weights()
    hidden_values, output_scores = forward(inputs, hidden_weights, output_weights)

    assert len(hidden_values) == NO_OF_HIDDEN_NEURONS
    assert len(output_scores) == NO_OF_TOKENS
    assert all(isinstance(v, float) for v in hidden_values)
    assert all(isinstance(v, float) for v in output_scores)


def test_forward_computes_correct_dot_products():
    inputs = [1, 2, 3]
    hidden_weights = [
        [1.0, 0.0, 0.0],  # picks out input[0]
        [0.0, 1.0, 1.0],  # sums input[1] + input[2]
    ]
    output_weights = [
        [1.0, 1.0],  # sums both hidden values
        [2.0, 0.0],  # doubles first hidden value only
    ]

    hidden_values, output_scores = forward(inputs, hidden_weights, output_weights)

    assert np.array_equal(hidden_values, [1.0, 5.0])  # [1*1, 2*1 + 3*1]
    assert np.array_equal(output_scores, [6.0, 2.0])  # [1+5, 2*1]


def test_forward_all_zero_inputs_produce_zero_scores():
    inputs = [0] * CONTEXT_WINDOW
    hidden_weights, output_weights = init_weights()

    hidden_values, output_scores = forward(inputs, hidden_weights, output_weights)

    assert np.array_equal(hidden_values, [0.0] * NO_OF_HIDDEN_NEURONS)
    assert np.array_equal(output_scores, [0.0] * NO_OF_TOKENS)


def test_forward_returns_hidden_values_and_output_scores_as_tuple():
    inputs = [1, 2, 3, 4, 5, 6, 7, 8]
    hidden_weights, output_weights = init_weights()
    result = forward(inputs, hidden_weights, output_weights)

    assert isinstance(result, tuple)
    assert len(result) == 2


# Equal scores -> softmax gives each of the 4 options a 0.25 probability,
# so the cumulative thresholds land at 0.25, 0.5, 0.75, 1.0.
EQUAL_SCORES = [0.0, 0.0, 0.0, 0.0]


@patch("src.network.random")
def test_predict_picks_first_bucket_for_low_sample(mock_random):
    mock_random.return_value = 0.1  # below cumulative 0.25
    assert predict(EQUAL_SCORES) == 0


@patch("src.network.random")
def test_predict_picks_second_bucket(mock_random):
    mock_random.return_value = 0.3  # between 0.25 and 0.5
    assert predict(EQUAL_SCORES) == 1


@patch("src.network.random")
def test_predict_picks_third_bucket(mock_random):
    mock_random.return_value = 0.6  # between 0.5 and 0.75
    assert predict(EQUAL_SCORES) == 2


@patch("src.network.random")
def test_predict_falls_back_to_last_bucket_when_sample_near_one(mock_random):
    # cumulative probability never exceeds a sample this close to 1.0,
    # so predict() should fall back to the last index.
    mock_random.return_value = 0.999999
    assert predict(EQUAL_SCORES) == 3


def test_predict_single_score_always_returns_zero_index():
    # only one possible outcome, regardless of the random sample drawn
    assert predict([0.0]) == 0
    assert predict([42.0]) == 0
    assert predict([-42.0]) == 0


def test_predict_returns_index_within_bounds_over_many_trials():
    scores = [0.1, -2.0, 3.0, 0.0, 1.5]
    for _ in range(200):
        prediction = predict(scores)
        assert isinstance(prediction, int)
        assert 0 <= prediction < len(scores)


def test_predict_strongly_favors_the_highest_score():
    # one score dominates, so softmax assigns it near-certain probability
    scores = [100.0, 0.0, 0.0, 0.0]
    predictions = [predict(scores) for _ in range(200)]
    assert predictions.count(0) == 200


def test_predict_with_real_forward_output_is_a_valid_token_index():
    inputs = [1, 2, 3, 4, 5, 6, 7, 8]
    hidden_weights, output_weights = init_weights()
    _, output_scores = forward(inputs, hidden_weights, output_weights)

    prediction = predict(output_scores)

    assert isinstance(prediction, int)
    assert 0 <= prediction < NO_OF_TOKENS
