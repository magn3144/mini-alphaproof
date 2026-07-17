from pathlib import Path
from typing import Callable

from alphaproof.core.environment import Environment
from alphaproof.core.paths import (
    DEFAULT_THEOREMS_PATH,
    LEAN_PROJECT_DIR,
    MODELS_DIR,
)
from leantree import LeanProject


DEFAULT_TOKENIZER_MODEL = str(MODELS_DIR / 'Salesforce--codet5p-220m')


class Config:
    """Hyperparameters and constructors used by the pseudocode pipeline."""

    def __init__(
        self,
        num_simulations: int,
        batch_size: int,
        num_actors: int = 1,
        num_games: int = 1,
        lr: float = 1e-5,
        environment_ctor: Callable[[], Environment] = (
            lambda: Environment(LeanProject(str(LEAN_PROJECT_DIR)))
        ),
        tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
        dataset_path: str | Path = DEFAULT_THEOREMS_PATH,
        run_id: int | str = 0,
        sft_run_dir: str | Path | None = None,
        max_state_length: int = 640,
        max_action_length: int = 128,
        training_steps: int = int(1000e3),
        checkpoint_interval: int = int(1e3),
        window_size: int = int(1e6),
        value_weight: float = 0.001,
        validation_fraction: float = 0.05,
        validation_batch_size: int = 64,
        validation_interval: int = 100,
        log_interval: int = 10,
        reward_window: int = 100,
        wandb_project: str = 'alphaproof',
        wandb_entity: str | None = None,
        wandb_tags: tuple[str, ...] = (),
    ):
        """Populate acting, search, training, and matchmaker settings."""
        ### Acting
        self.environment_ctor = environment_ctor
        self.dataset_path = Path(dataset_path)
        self.sft_run_dir = Path(sft_run_dir) if sft_run_dir is not None else None
        if self.sft_run_dir is None:
            self.tokenizer_model = tokenizer_model
            self.initial_params_path = None
        else:
            self.tokenizer_model = str(self.sft_run_dir / 'model_source')
            self.initial_params_path = self.sft_run_dir / 'network_params.pt'
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
        self.training_steps = training_steps
        self.checkpoint_interval = checkpoint_interval
        self.window_size = window_size
        self.batch_size = batch_size
        self.max_state_length = max_state_length
        self.max_action_length = max_action_length
        self.lr = lr
        self.value_weight = value_weight
        self.validation_fraction = validation_fraction
        self.validation_batch_size = validation_batch_size
        self.validation_interval = validation_interval
        self.log_interval = log_interval
        self.reward_window = reward_window

        ### Logging
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_tags = wandb_tags

        # Matchmaker
        self.mm_disprove_rate = 0.5
        self.mm_trust_count = 8
        self.mm_fully_decided_trust_count = 12
        self.mm_proved_weight = 1e-3
        self.mm_undecided_weight = 0.1
        self.mm_simulation_failure_multiplier = 2.0
        self.mm_max_num_simulations = 16 * num_simulations

        self.run_id = run_id
