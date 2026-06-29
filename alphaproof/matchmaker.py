import dataclasses
import random

from alphaproof.config import Config
from alphaproof.environment import Theorem
from alphaproof.game import Game


class Matchmaker:
    """Chooses theorem objectives and records their outcomes."""

    @dataclasses.dataclass
    class Stats:
        """Statistics for a theorem."""
        # List of (disprove, result) tuples:
        # Disprove is True iff this was an attempt to disprove the theorem.
        # Result is True iff the attempt was successful.
        attempts: list[tuple[bool, bool]]

        def update(self, game: Game):
            """Update statistics with the results of a game."""
            self.attempts.append((game.disprove, game.root.is_optimal))

        def weight(self, config: Config) -> float:
            """Compute weight of this theorem."""
            if not self.attempts:
                return 1.0
            disproved = any(
                    disprove and success for (disprove, success) in self.attempts
            )
            proved = any(
                    (not disprove) and success for (disprove, success) in self.attempts
            )
            if disproved:
                return 0.0
            elif len(self.attempts) < config.mm_trust_count:
                return 1.0
            elif not disproved and not proved:
                # Never managed to prove or disprove.
                return config.mm_undecided_weight
            else:
                latest = self.attempts[-config.mm_fully_decided_trust_count :]
                if all((not disprove) and success for (disprove, success) in latest):
                    # Consistently proved.
                    return config.mm_proved_weight
            return 1.0

    def __init__(self, config: Config):
        """Initialize theorem statistics from the backing store."""
        self.config = config
        # Load theorems and their stats from the database.
        self.theorem_stats: dict[Theorem, Matchmaker.Stats] = {}

    def compute_num_simulations(self, theorem: Theorem, stats: Stats) -> int:
        """Compute number of simulations to run for a theorem."""
        return 1000

    def get_start_position(self) -> Game:
        """Get a start position for a new game to be played."""
        # Get a theorem to be proved or disproved based on the per-theorem stats.
        # Prefer interesting theorems.
        weights = [
                stats.weight(self.config) for stats in self.theorem_stats.values()
        ]
        [(theorem, stats)] = random.choices(
                list(self.theorem_stats.items()), weights, k=1
        )
        disprove = random.random() < self.config.mm_disprove_rate
        num_simulations = self.compute_num_simulations(theorem, stats)
        return Game(
                theorem=theorem, disprove=disprove, num_simulations=num_simulations
        )

    def send_game(self, game: Game):
        """Send completed game to matchmaker."""
        self.theorem_stats[game.theorem].update(game)
