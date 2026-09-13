"""View helper of the SSD games: a square window of the map around a position, padded with ``'0'``.

Vendored from ``utility_funcs.py`` of https://github.com/011235813/sequential_social_dilemma_games (MIT
licence, see LICENSE and NOTICE in this directory); the video helpers are dropped and the padding
character is explicit (``'0'``, the "beyond the map" cell of the SSD colour map).
"""
import numpy as np

PAD_CHAR = "0"


def return_view(grid: np.ndarray, pos, row_size: int, col_size: int) -> np.ndarray:
    """``(2 * row_size + 1) x (2 * col_size + 1)`` window of ``grid`` centred on ``pos`` (row, col).

    Cells outside the map are ``'0'``.  (The SSD source names the axes x / y and pads with the integer 0,
    which numpy turns into the string ``'0'`` for character arrays; the result is identical.)
    """
    r, c = int(pos[0]), int(pos[1])
    padded = np.pad(grid, ((row_size, row_size), (col_size, col_size)), constant_values=PAD_CHAR)
    r += row_size
    c += col_size
    return padded[r - row_size:r + row_size + 1, c - col_size:c + col_size + 1]
