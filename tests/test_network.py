from src.network import (
    CONTEXT_WINDOW,
    MAX_WEIGHT,
    MIN_WEIGHT,
    NO_OF_HIDDEN_NEURONS,
    NO_OF_TOKENS,
    gen_weights,
    init_weights,
)


def test_init_weights_returns_two_lists():
    result = init_weights()
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_hidden_weights_shape():
    hidden_weights, _ = init_weights()
    assert len(hidden_weights) == NO_OF_HIDDEN_NEURONS
    assert len(hidden_weights) == 64
    for neuron_weights in hidden_weights:
        assert isinstance(neuron_weights, list)
        assert len(neuron_weights) == CONTEXT_WINDOW
        assert len(neuron_weights) == 8


def test_output_weights_shape():
    _, output_weights = init_weights()
    assert len(output_weights) == NO_OF_TOKENS
    assert len(output_weights) == 28
    for neuron_weights in output_weights:
        assert isinstance(neuron_weights, list)
        assert len(neuron_weights) == NO_OF_HIDDEN_NEURONS
        assert len(neuron_weights) == 64


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
    assert first_hidden != second_hidden
    assert first_output != second_output


def test_gen_weights_returns_requested_count_within_bounds():
    weights = gen_weights(10)
    assert len(weights) == 10
    assert all(MIN_WEIGHT <= w <= MAX_WEIGHT for w in weights)


def test_gen_weights_zero_length():
    assert gen_weights(0) == []
