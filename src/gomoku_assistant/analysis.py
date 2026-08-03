from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .domain import BoardState, FOUR_DIRECTIONS, Stone


class ProofStatus(str, Enum):
    HEURISTIC = "heuristic"
    WIN_IN_ONE = "win-in-one"
    BLOCK_REQUIRED = "block-required"
    FORCED_WIN = "forced-win"
    FORCED_LOSS = "forced-loss"


@dataclass(frozen=True)
class CandidateMove:
    x: int
    y: int
    rank: int
    score: int | None
    proof: ProofStatus
    principal_variation: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class SearchStats:
    """Metadata reported by an engine search, not an engine evaluation."""

    requested_time_ms: int
    elapsed_ms: int
    engine_time_ms: int | None = None
    depth: str | None = None
    threads: int | None = None
    hash_kib: int | None = None


@dataclass(frozen=True)
class AnalysisResult:
    board: BoardState
    candidates: tuple[CandidateMove, ...]
    engine_name: str
    raw_output: str = ""
    search_stats: SearchStats | None = None


class HeuristicAnalyzer:
    """Fast local fallback. Strong engine analysis is delegated to Rapfi."""

    name = "Local tactical heuristic"

    def analyze(self, board: BoardState, limit: int = 3) -> AnalysisResult:
        if not board.is_count_legal() or board.is_terminal():
            return AnalysisResult(board=board, candidates=(), engine_name=self.name)

        side = board.side_to_move()
        opponent = side.opponent
        candidates: list[CandidateMove] = []
        opponent_wins = {
            point
            for point in self._candidate_points(board)
            if board.place(*point, stone=opponent).winner() is opponent
        }

        for x, y in self._candidate_points(board):
            own_after = board.place(x, y, stone=side)
            if own_after.winner() is side:
                score = 100_000_000
                proof = ProofStatus.WIN_IN_ONE
            else:
                attack = self._move_strength(board, x, y, side)
                defense = self._move_strength(board, x, y, opponent)
                score = attack + int(defense * 0.92)
                proof = ProofStatus.BLOCK_REQUIRED if (x, y) in opponent_wins else ProofStatus.HEURISTIC
            candidates.append(
                CandidateMove(
                    x=x,
                    y=y,
                    rank=0,
                    score=score,
                    proof=proof,
                    principal_variation=((x, y),),
                )
            )

        candidates.sort(key=lambda candidate: (candidate.score, -candidate.y, -candidate.x), reverse=True)
        ranked = tuple(
            CandidateMove(
                x=candidate.x,
                y=candidate.y,
                rank=index,
                score=candidate.score,
                proof=candidate.proof,
                principal_variation=candidate.principal_variation,
            )
            for index, candidate in enumerate(candidates[:limit], start=1)
        )
        return AnalysisResult(board=board, candidates=ranked, engine_name=self.name)

    def _candidate_points(self, board: BoardState) -> tuple[tuple[int, int], ...]:
        occupied = [
            (x, y)
            for y in range(board.size)
            for x in range(board.size)
            if board.at(x, y) is not Stone.EMPTY
        ]
        if not occupied:
            center = board.size // 2
            return ((center, center),)

        points: set[tuple[int, int]] = set()
        for stone_x, stone_y in occupied:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    x, y = stone_x + dx, stone_y + dy
                    if board.in_bounds(x, y) and board.at(x, y) is Stone.EMPTY:
                        points.add((x, y))
        return tuple(sorted(points, key=lambda point: (point[1], point[0])))

    def _move_strength(self, board: BoardState, x: int, y: int, stone: Stone) -> int:
        score = 0
        for dx, dy in FOUR_DIRECTIONS:
            left_count, left_open = self._count_direction(board, x, y, -dx, -dy, stone)
            right_count, right_open = self._count_direction(board, x, y, dx, dy, stone)
            total = 1 + left_count + right_count
            open_ends = int(left_open) + int(right_open)
            score += self._pattern_score(total, open_ends)
        return score

    @staticmethod
    def _count_direction(
        board: BoardState, x: int, y: int, dx: int, dy: int, stone: Stone
    ) -> tuple[int, bool]:
        count = 0
        x += dx
        y += dy
        while board.in_bounds(x, y) and board.at(x, y) is stone:
            count += 1
            x += dx
            y += dy
        return count, board.in_bounds(x, y) and board.at(x, y) is Stone.EMPTY

    @staticmethod
    def _pattern_score(total: int, open_ends: int) -> int:
        if total >= 5:
            return 100_000_000
        if total == 4 and open_ends == 2:
            return 2_000_000
        if total == 4 and open_ends == 1:
            return 200_000
        if total == 3 and open_ends == 2:
            return 25_000
        if total == 3 and open_ends == 1:
            return 2_500
        if total == 2 and open_ends == 2:
            return 500
        if total == 2 and open_ends == 1:
            return 80
        return 8
