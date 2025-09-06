from __future__ import annotations

from typing import List


def rle_encode(tokens: List[List[int]]) -> List:
    """Run-length encode a sequence of token lists."""
    if not tokens:
        return []
    encoded: List = []
    prev = tokens[0]
    count = 1
    for curr in tokens[1:]:
        if curr == prev:
            count += 1
        else:
            if count > 1:
                encoded.append(["NC", count])
            else:
                encoded.append(prev)
            prev = curr
            count = 1
    if count > 1:
        encoded.append(["NC", count])
    else:
        encoded.append(prev)
    return encoded
