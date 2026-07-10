import string

# Let's just map out only lowercase alphas
char_to_idx: dict[str] = {
    char: i for i, char in enumerate(string.ascii_lowercase + " " + "-")
}

# Also map out each number to lowercase alphas
idx_to_char: dict[int] = {
    i: char for i, char in enumerate(string.ascii_lowercase + " " + "-")
}


def encode(text: str) -> list[float]:
    text = text[:8]  # strictly keep the input size to 8
    return [char_to_idx[char] for char in text]


def decode(tokens: list[int]) -> str:
    output: str = ""
    for token in tokens:
        output += idx_to_char[token]
    return output
