import torch
from torch.optim import RMSprop
from torch.nn import LeakyReLU

from maro.rl.exploration import MultiLinearExplorationScheduler, epsilon_greedy
from maro.rl.model import DiscreteQNet
from maro.rl.policy import ValueBasedPolicy
from maro.rl.training.algorithms import DQNParams, DQNTrainer

from .networks import QuantumFullyConnected

q_net_conf = {
    "hidden_dims": [256, 128, 64, 32],
    "n_qubits": 4,
    "n_layers": 6,
    "activation": LeakyReLU,
    "softmax": False,
    "batch_norm": True,
    "skip_connection": False,
    "head": True,
    "dropout_p": 0.0,
}
learning_rate = 0.05


class QuantumQNet(DiscreteQNet):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(QuantumQNet, self).__init__(state_dim=state_dim, action_num=action_num)
        self._fc = QuantumFullyConnected(
            input_dim=state_dim, output_dim=action_num, **q_net_conf)
        self._optim = RMSprop(self._fc.parameters(), lr=learning_rate)

    def _get_q_values_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        return self._fc(states.float())


def get_quantum_dqn_policy(state_dim: int, action_num: int, name: str) -> ValueBasedPolicy:
    return ValueBasedPolicy(
        name=name,
        q_net=QuantumQNet(state_dim, action_num),
        exploration_strategy=(epsilon_greedy, {"epsilon": 0.4}),
        exploration_scheduling_options=[
            (
                "epsilon",
                MultiLinearExplorationScheduler,
                {
                    "splits": [(2, 0.32)],
                    "initial_value": 0.4,
                    "last_ep": 5,
                    "final_value": 0.0,
                },
            ),
        ],
        warmup=100,
    )


def get_quantum_dqn(name: str) -> DQNTrainer:
    return DQNTrainer(
        name=name,
        params=DQNParams(
            reward_discount=0.0,
            update_target_every=5,
            num_epochs=10,
            soft_update_coef=0.1,
            double=False,
            replay_memory_capacity=10000,
            random_overwrite=False,
            batch_size=32,
        ),
    )