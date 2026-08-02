from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable


class Stone(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self) -> "Stone":
        if self is Stone.BLACK:
            return Stone.WHITE
        if self is Stone.WHITE:
            return Stone.BLACK
        raise ValueError("Empty does not have an opponent.")


Direction = tuple[int, int]
FOUR_DIRECTIONS: tuple[Direction, ...] = ((1, 0), (0, 1), (1, 1), (1, -1))


@dataclass(frozen=True)
class WinningLine:
    stone: Stone
    points: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BoardState:
    """Immutable square freestyle Gomoku board."""

    size: int = 15
    cells: tuple[Stone, ...] = field(default_factory=lambda: (Stone.EMPTY,) * 225)

    def __post_init__(self) -> None:
        if self.size < 5:
            raise ValueError("Board size must be at least 5.")
        if len(self.cells) != self.size * self.size:
            raise ValueError("Cell count does not match board size.")
        if any(not isinstance(cell, Stone) for cell in self.cells):
            raise TypeError("Cells must contain Stone values.")

    @classmethod
    def empty(cls, size: int = 15) -> "BoardState":
        return cls(size=size, cells=(Stone.EMPTY,) * (size * size))

    @classmethod
    def from_rows(cls, rows: Iterable[Iterable[int | Stone]]) -> "BoardState":
        normalized = tuple(tuple(Stone(value) for value in row) for row in rows)
        if not normalized or any(len(row) != len(normalized) for row in normalized):
            raise ValueError("Rows must describe a non-empty square board.")
        return cls(size=len(normalized), cells=tuple(cell for row in normalized for cell in row))

    def to_rows(self) -> tuple[tuple[Stone, ...], ...]:
        return tuple(
            self.cells[row * self.size : (row + 1) * self.size] for row in range(self.size)
        )

    def index(self, x: int, y: int) -> int:
        if not self.in_bounds(x, y):
            raise IndexError(f"Point outside board: {x},{y}")
        return y * self.size + x

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def at(self, x: int, y: int) -> Stone:
        return self.cells[self.index(x, y)]

    def set_cell(self, x: int, y: int, stone: Stone) -> "BoardState":
        cells = list(self.cells)
        cells[self.index(x, y)] = Stone(stone)
        return BoardState(size=self.size, cells=tuple(cells))

    def place(self, x: int, y: int, stone: Stone | None = None) -> "BoardState":
        if self.at(x, y) is not Stone.EMPTY:
            raise ValueError(f"Point is occupied: {x},{y}")
        move_stone = stone or self.side_to_move()
        if move_stone is Stone.EMPTY:
            raise ValueError("Cannot place an empty stone.")
        return self.set_cell(x, y, move_stone)

    def counts(self) -> tuple[int, int]:
        return self.cells.count(Stone.BLACK), self.cells.count(Stone.WHITE)

    def is_count_legal(self) -> bool:
        black, white = self.counts()
        return black == white or black == white + 1

    def side_to_move(self) -> Stone:
        black, white = self.counts()
        if black == white:
            return Stone.BLACK
        if black == white + 1:
            return Stone.WHITE
        raise ValueError("Stone counts are not legal for black-first Gomoku.")

    def legal_moves(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (x, y)
            for y in range(self.size)
            for x in range(self.size)
            if self.at(x, y) is Stone.EMPTY
        )

    def winner(self) -> Stone | None:
        line = self.winning_line()
        return line.stone if line else None

    def winning_line(self) -> WinningLine | None:
        for y in range(self.size):
            for x in range(self.size):
                stone = self.at(x, y)
                if stone is Stone.EMPTY:
                    continue
                for dx, dy in FOUR_DIRECTIONS:
                    previous_x, previous_y = x - dx, y - dy
                    if self.in_bounds(previous_x, previous_y) and self.at(previous_x, previous_y) is stone:
                        continue
                    points: list[tuple[int, int]] = []
                    cursor_x, cursor_y = x, y
                    while self.in_bounds(cursor_x, cursor_y) and self.at(cursor_x, cursor_y) is stone:
                        points.append((cursor_x, cursor_y))
                        cursor_x += dx
                        cursor_y += dy
                    if len(points) >= 5:
                        return WinningLine(stone=stone, points=tuple(points))
        return None

    def is_terminal(self) -> bool:
        return self.winner() is not None or not self.legal_moves()

    def coordinate_name(self, x: int, y: int) -> str:
        return f"{chr(ord('A') + x)}{y + 1}"


@dataclass(frozen=True)
class TransitionResult:
    valid: bool
    changed: bool
    reason: str = ""
    added_count: int = 0


def validate_transition(previous: BoardState | None, current: BoardState) -> TransitionResult:
    """Validate a visual state transition without attempting to repair it."""

    if previous is None:
        if not current.is_count_legal():
            return TransitionResult(False, False, "black/white counts are invalid")
        return TransitionResult(True, True, "initial state", sum(current.counts()))

    if previous.size != current.size:
        return TransitionResult(False, False, "board size changed")
    if previous.is_terminal() and previous.cells != current.cells:
        return TransitionResult(False, False, "game was already terminal")

    changes = [
        (x, y, previous.at(x, y), current.at(x, y))
        for y in range(current.size)
        for x in range(current.size)
        if previous.at(x, y) is not current.at(x, y)
    ]
    if not changes:
        return TransitionResult(True, False, "unchanged")

    if any(before is not Stone.EMPTY or after is Stone.EMPTY for _, _, before, after in changes):
        return TransitionResult(False, False, "a previous stone was removed or recolored")

    try:
        expected = previous.side_to_move()
    except ValueError:
        return TransitionResult(False, False, "previous board has invalid stone counts")
    added_count = len(changes)
    black_added = sum(after is Stone.BLACK for _, _, _, after in changes)
    white_added = added_count - black_added
    expected_black = (added_count + int(expected is Stone.BLACK)) // 2
    expected_white = added_count - expected_black
    if black_added != expected_black or white_added != expected_white:
        return TransitionResult(False, False, "new stones do not follow turn order")
    if not current.is_count_legal():
        return TransitionResult(False, False, "black/white counts are invalid")
    if added_count == 1:
        return TransitionResult(True, True, "legal placement", added_count)
    return TransitionResult(True, True, f"caught up {added_count} placements", added_count)
