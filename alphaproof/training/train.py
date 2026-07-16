from alphaproof.core.actors import run_actor
from alphaproof.core.config import Config
from alphaproof.training.matchmaker import Matchmaker
from alphaproof.core.network import Network
from alphaproof.training.replay_buffer import ReplayBuffer
from alphaproof.training.shared_storage import SharedStorage


def train_network(
    config: Config,
    storage: SharedStorage,
    replay_buffer: ReplayBuffer,
):
    """Train the network from replay and checkpoint parameters."""
    network = Network(config)
    network.params = storage.latest_params()

    if len(replay_buffer) == 0:
        print('Warning: replay buffer is empty; skipping network training.')
        return

    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_params(i, network.params)
        batch = replay_buffer.sample_batch()
        network.update(batch)
    storage.save_params(config.training_steps, network.params)


def launch_job(f, *args):
    """Launch a worker job in the pseudocode runtime."""
    f(*args)


# AlphaProof training is split into two independent parts: A learner which
# updates the network, and actors which play games to generate data.
# These two parts only communicate by transferring the latest network checkpoint
# from the learner to the actor, and the finished games from the actor
# to the learner.
def alphaproof_train(config: Config) -> Network:
    """Coordinate actor jobs and learner updates for AlphaProof training."""
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)
    matchmaker = Matchmaker(config)

    network = Network(config)
    if config.initial_params_path is not None:
        network.load_params(config.initial_params_path)
    storage.save_params(0, network.params)

    for _ in range(config.num_actors):
        launch_job(
                run_actor,
                config,
                storage,
                replay_buffer,
                matchmaker,
                config.num_games,
        )

    train_network(config, storage, replay_buffer)

    return storage.latest_params()
