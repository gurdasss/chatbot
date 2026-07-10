from src.codec import encode, decode, char_to_idx, idx_to_char


def test_encode_text():
    text: str = "hello"
    assert encode(text) == [char_to_idx[char] for char in text]


def test_decode_tokens():
    text: str = "hello"
    tokens: list[int] = [char_to_idx[char] for char in text]
    assert decode(tokens) == text
