import typing
from typing import Any, Callable, List, Dict

from alphaproof.core.environment import Environment
from alphaproof.core.paths import LEAN_PROJECT_DIR, MODELS_DIR
from leantree import LeanProject, LeanTactic, LeanProofState


DEFAULT_TOKENIZER_MODEL = str(MODELS_DIR / 'Salesforce--codet5p-220m')


class Config:
    """Hyperparameters and constructors used by the pseudocode pipeline."""

    def __init__(
        self,
        num_simulations: int,
        batch_size: int,
        num_actors: int,
        num_games: int,
        lr: float,
        environment_ctor: Callable[[], Environment] = (
            lambda: Environment(LeanProject(str(LEAN_PROJECT_DIR)))
        ),
        tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
    ):
        """Populate acting, search, training, and matchmaker settings."""
        ### Acting
        self.environment_ctor = environment_ctor
        self.tokenizer_model = tokenizer_model
        self.num_actors = num_actors
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.tactic_timeout = 1.0

        # UCB formula
        self.pb_c_base = 3200
        self.pb_c_init = 0.001
        self.value_discount = 0.99
        self.prior_temperature = 200

        # Other MCTS parameters
        self.no_legal_actions_value = -40

        # Progressive sampling parameters
        self.ps_c = 0.01
        self.ps_alpha = 0.6

        # Value predictions
        self.num_value_bins = 64

        ### Training
        self.training_steps = int(1000e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.sequence_length = 32
        self.lr = lr
        self.value_weight = 0.001

        # Matchmaker
        self.mm_disprove_rate = 0.5
        self.mm_trust_count = 8
        self.mm_fully_decided_trust_count = 12
        self.mm_proved_weight = 1e-3
        self.mm_undecided_weight = 0.1
        self.mm_simulation_failure_multiplier = 2.0
        self.mm_max_num_simulations = 16 * num_simulations

        # Run id
        # TODO: Update this when the game starts
        self.run_id = 0
