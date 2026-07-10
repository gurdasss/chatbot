import pytest

from src.codec import char_to_idx, decode, encode
from src.dataset import CONTEXT_WINDOW, make_pairs


def test_make_pairs_example_from_spec():
    text = "how are you"
    pairs = make_pairs(text)

    assert pairs[0] == (encode("how are "), char_to_idx["y"])
    assert pairs[1] == (encode("ow are y"), char_to_idx["o"])
    assert pairs[2] == (encode("w are yo"), char_to_idx["u"])


def test_make_pairs_count_matches_sliding_window():
    text = "how are you"  # 11 chars
    pairs = make_pairs(text)
    # one pair per starting position where 8 input + 1 output chars fit
    assert len(pairs) == len(text) - CONTEXT_WINDOW
    assert len(pairs) == 3


def test_make_pairs_shape_of_each_pair():
    pairs = make_pairs("how are you")
    for inputs, output in pairs:
        assert isinstance(inputs, list)
        assert len(inputs) == CONTEXT_WINDOW
        assert all(isinstance(token, int) for token in inputs)
        assert isinstance(output, int)


def test_make_pairs_reconstructs_original_text():
    text = "how are you"
    pairs = make_pairs(text)
    for i, (inputs, output) in enumerate(pairs):
        window = text[i : i + CONTEXT_WINDOW]
        next_char = text[i + CONTEXT_WINDOW]
        assert inputs == encode(window)
        assert output == char_to_idx[next_char]
        assert decode(inputs) + decode([output]) == text[i : i + CONTEXT_WINDOW + 1]


def test_make_pairs_empty_text_returns_no_pairs():
    assert make_pairs("") == []


def test_make_pairs_text_shorter_than_window_plus_one_returns_no_pairs():
    assert make_pairs("hi") == []
    assert make_pairs("a" * CONTEXT_WINDOW) == []  # exactly 8 chars, no output char left


def test_make_pairs_text_exactly_window_plus_one_returns_single_pair():
    text = "a" * CONTEXT_WINDOW + "b"  # 9 chars
    pairs = make_pairs(text)
    assert len(pairs) == 1
    assert pairs[0] == (encode("a" * CONTEXT_WINDOW), char_to_idx["b"])


def test_make_pairs_unsupported_char_raises_key_error():
    with pytest.raises(KeyError):
        make_pairs("How are you")  # uppercase 'H' not in vocab
