from src.codec import encode

CONTEXT_WINDOW: int = 8


def make_pairs(text: str) -> list[tuple[list[int], int]]:
    data: list[tuple[list[int], int]] = []

    for i in range(len(text)):
        text_to_encode: str = text[i : i + CONTEXT_WINDOW + 1]  # [i : i + 9)
        if len(text_to_encode) < CONTEXT_WINDOW + 1:  # 8 inputs + 1 output
            break
        encoded_inputs_and_output: list[int] = encode(text_to_encode)

        data.append(
            (encoded_inputs_and_output[:CONTEXT_WINDOW], encoded_inputs_and_output[-1])
        )

    return data
