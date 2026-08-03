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


class RecommendationMode(str, Enum):
    """How much tactical freedom the side to move actually has."""

    NORMAL = "normal"
    WIN_NOW = "win-now"
    FORCED_DEFENSE = "forced-defense"
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
class RejectedMove:
    x: int
    y: int
    source: str
    reason: str
    danger_points: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class AnalysisResult:
    board: BoardState
    candidates: tuple[CandidateMove, ...]
    engine_name: str
    raw_output: str = ""
    search_stats: SearchStats | None = None
    recommendation_mode: RecommendationMode = RecommendationMode.NORMAL
    danger_points: tuple[tuple[int, int], ...] = ()
    safe_candidate_count: int = 0
    rejected_moves: tuple[RejectedMove, ...] = ()


@dataclass(frozen=True)
class TacticalAssessment:
    mode: RecommendationMode
    candidates: tuple[CandidateMove, ...] = ()
    danger_points: tuple[tuple[int, int], ...] = ()


def candidate_points(board: BoardState, radius: int = 2) -> tuple[tuple[int, int], ...]:
    """Return empty points near play; all one-move wins are in this set."""

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
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = stone_x + dx, stone_y + dy
                if board.in_bounds(x, y) and board.at(x, y) is Stone.EMPTY:
                    points.add((x, y))
    return tuple(sorted(points, key=lambda point: (point[1], point[0])))


def is_winning_move(board: BoardState, x: int, y: int, stone: Stone) -> bool:
    """Check a prospective move using only the four lines through that point."""

    if stone is Stone.EMPTY or not board.in_bounds(x, y) or board.at(x, y) is not Stone.EMPTY:
        return False
    for dx, dy in FOUR_DIRECTIONS:
        total = 1
        for direction_x, direction_y in ((dx, dy), (-dx, -dy)):
            cursor_x, cursor_y = x + direction_x, y + direction_y
            while board.in_bounds(cursor_x, cursor_y) and board.at(cursor_x, cursor_y) is stone:
                total += 1
                cursor_x += direction_x
                cursor_y += direction_y
        if total >= 5:
            return True
    return False


def immediate_winning_points(
    board: BoardState,
    stone: Stone,
    points: tuple[tuple[int, int], ...] | None = None,
) -> tuple[tuple[int, int], ...]:
    choices = candidate_points(board) if points is None else points
    return tuple(point for point in choices if is_winning_move(board, *point, stone))


def assess_tactical_position(board: BoardState, limit: int = 3) -> TacticalAssessment:
    """Resolve wins and mandatory one-ply defenses before engine evaluation."""

    if not board.is_count_legal() or board.is_terminal():
        return TacticalAssessment(RecommendationMode.NORMAL)
    side = board.side_to_move()
    own_wins = immediate_winning_points(board, side)
    if own_wins:
        moves = tuple(
            CandidateMove(x, y, rank, 100_000_000, ProofStatus.WIN_IN_ONE, ((x, y),))
            for rank, (x, y) in enumerate(own_wins[:limit], start=1)
        )
        return TacticalAssessment(RecommendationMode.WIN_NOW, moves)

    danger_points = immediate_winning_points(board, side.opponent)
    if len(danger_points) == 1:
        x, y = danger_points[0]
        return TacticalAssessment(
            RecommendationMode.FORCED_DEFENSE,
            (CandidateMove(x, y, 1, None, ProofStatus.BLOCK_REQUIRED, ((x, y),)),),
            danger_points,
        )
    if len(danger_points) >= 2:
        return TacticalAssessment(RecommendationMode.FORCED_LOSS, (), danger_points)
    return TacticalAssessment(RecommendationMode.NORMAL)


def validate_candidate_safety(
    board: BoardState, point: tuple[int, int]
) -> tuple[str, tuple[tuple[int, int], ...]] | None:
    """Reject moves that lose immediately or permit an unanswered double threat."""

    x, y = point
    side = board.side_to_move()
    if not board.in_bounds(x, y) or board.at(x, y) is not Stone.EMPTY:
        return "not a legal empty point", ()
    after_move = board.place(x, y, side)
    if is_winning_move(board, x, y, side):
        return None

    immediate_dangers = immediate_winning_points(after_move, side.opponent)
    if immediate_dangers:
        return "allows an immediate opponent win", immediate_dangers

    # An opponent reply which leaves two endpoints is normally decisive. A direct
    # counter-win remains valid, so do not discard that tactical race.
    for reply in candidate_points(after_move):
        after_reply = after_move.place(*reply, side.opponent)
        if immediate_winning_points(after_reply, side):
            continue
        threats = immediate_winning_points(after_reply, side.opponent)
        if len(threats) >= 2:
            return "allows an opponent double threat", threats
    return None


def filter_safe_candidates(
    board: BoardState,
    ranked_moves: tuple[tuple[str, CandidateMove], ...],
    limit: int = 3,
) -> tuple[tuple[CandidateMove, ...], tuple[RejectedMove, ...]]:
    """Keep only legal moves that survive the deterministic tactical checks."""

    accepted: list[CandidateMove] = []
    rejected: list[RejectedMove] = []
    seen: set[tuple[int, int]] = set()
    for source, move in ranked_moves:
        point = (move.x, move.y)
        if point in seen:
            continue
        seen.add(point)
        safety = validate_candidate_safety(board, point)
        if safety is not None:
            reason, danger_points = safety
            rejected.append(RejectedMove(move.x, move.y, source, reason, danger_points))
            continue
        accepted.append(move)
        if len(accepted) >= limit:
            break
    return (
        tuple(
            CandidateMove(
                move.x, move.y, rank, move.score, move.proof, move.principal_variation
            )
            for rank, move in enumerate(accepted, start=1)
        ),
        tuple(rejected),
    )


class HeuristicAnalyzer:
    """Fast local fallback. Strong engine analysis is delegated to Rapfi."""

    name = "Local tactical heuristic"

    def analyze(self, board: BoardState, limit: int = 3) -> AnalysisResult:
        if not board.is_count_legal() or board.is_terminal():
            return AnalysisResult(board=board, candidates=(), engine_name=self.name)

        tactical = assess_tactical_position(board, limit)
        if tactical.mode is not RecommendationMode.NORMAL:
            return AnalysisResult(
                board=board,
                candidates=tactical.candidates,
                engine_name=self.name,
                recommendation_mode=tactical.mode,
                danger_points=tactical.danger_points,
                safe_candidate_count=len(tactical.candidates),
            )

        ranked = self.ranked_moves(board)
        candidates, rejected = filter_safe_candidates(
            board, tuple(("local", move) for move in ranked), limit
        )
        return AnalysisResult(
            board=board,
            candidates=candidates,
            engine_name=self.name,
            safe_candidate_count=len(candidates),
            rejected_moves=rejected,
        )

    def ranked_moves(self, board: BoardState) -> tuple[CandidateMove, ...]:
        """Score local choices without claiming that every high score is safe."""

        side = board.side_to_move()
        opponent = side.opponent
        candidates: list[CandidateMove] = []

        for x, y in candidate_points(board):
            if is_winning_move(board, x, y, side):
                score = 100_000_000
                proof = ProofStatus.WIN_IN_ONE
            else:
                attack = self._move_strength(board, x, y, side)
                defense = self._move_strength(board, x, y, opponent)
                score = attack + int(defense * 0.92)
                proof = ProofStatus.HEURISTIC
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
        return tuple(
            CandidateMove(
                x=candidate.x,
                y=candidate.y,
                rank=index,
                score=candidate.score,
                proof=candidate.proof,
                principal_variation=candidate.principal_variation,
            )
            for index, candidate in enumerate(candidates, start=1)
        )

    def _candidate_points(self, board: BoardState) -> tuple[tuple[int, int], ...]:
        return candidate_points(board)

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
