import string

import pytest

from src.codec import encode, decode, char_to_idx, idx_to_char


def test_encode_text():
    text: str = "hello"
    assert encode(text) == [char_to_idx[char] for char in text]


def test_decode_tokens():
    text: str = "hello"
    tokens: list[int] = [char_to_idx[char] for char in text]
    assert decode(tokens) == text


def test_vocab_size_and_indices():
    # 26 lowercase letters + space + hyphen
    assert len(char_to_idx) == 28
    assert char_to_idx["a"] == 0
    assert char_to_idx["z"] == 25
    assert char_to_idx[" "] == 26
    assert char_to_idx["-"] == 27


def test_char_to_idx_and_idx_to_char_are_inverses():
    for char, idx in char_to_idx.items():
        assert idx_to_char[idx] == char
    for idx, char in idx_to_char.items():
        assert char_to_idx[char] == idx


def test_encode_empty_string():
    assert encode("") == []


def test_decode_empty_list():
    assert decode([]) == ""


@pytest.mark.parametrize("char", list(string.ascii_lowercase) + [" ", "-"])
def test_encode_each_valid_char(char):
    assert encode(char) == [char_to_idx[char]]


def test_encode_word_with_space_and_hyphen():
    text = "co-op is"
    assert encode(text) == [char_to_idx[char] for char in text]


@pytest.mark.skip(reason="For now, decided to allow the encoding of 8+ chars")
def test_encode_truncates_to_eight_chars():
    text = "abcdefghij"  # 10 chars, only first 8 should be kept
    assert encode(text) == [char_to_idx[char] for char in text[:8]]
    assert len(encode(text)) == 8


def test_encode_exactly_eight_chars_not_truncated():
    text = "abcdefgh"
    assert encode(text) == [char_to_idx[char] for char in text]
    assert len(encode(text)) == 8


@pytest.mark.skip(reason="For now, decided to allow the encoding of 8+ chars")
def test_encode_nine_chars_drops_last_char():
    text = "abcdefghi"
    assert encode(text) == encode(text[:8])
    assert "i" not in decode(encode(text))


def test_encode_uppercase_raises_key_error():
    with pytest.raises(KeyError):
        encode("Hello")


@pytest.mark.parametrize("char", ["1", "!", ".", "_", "@", "\n", "\t"])
def test_encode_unsupported_char_raises_key_error(char):
    with pytest.raises(KeyError):
        encode(char)


def test_decode_out_of_range_token_raises_key_error():
    with pytest.raises(KeyError):
        decode([28])


def test_decode_negative_token_raises_key_error():
    with pytest.raises(KeyError):
        decode([-1])


def test_round_trip_encode_decode_within_limit():
    text = "hi you"
    assert decode(encode(text)) == text


@pytest.mark.skip(reason="For now, decided to allow the encoding of 8+ chars")
def test_round_trip_truncated_input_does_not_match_original():
    text = "hello world"  # longer than 8 chars
    assert decode(encode(text)) == text[:8]
    assert decode(encode(text)) != text
