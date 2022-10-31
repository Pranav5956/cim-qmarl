from functools import partial
from typing import Any, Callable, Dict, Optional

from maro.rl.policy import AbsPolicy
from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.rollout import AbsEnvSampler
from maro.rl.training import AbsTrainer

from .algorithms import creators
from .config import action_num, algorithm, env_conf, num_agents, reward_shaping_conf, state_dim
from .env_sampler import CIMEnvSampler


class CIMBundle(RLComponentBundle):
    def get_env_config(self) -> dict:
        return env_conf

    def get_test_env_config(self) -> Optional[dict]:
        return None

    def get_env_sampler(self) -> AbsEnvSampler:
        return CIMEnvSampler(self.env, self.test_env, reward_eval_delay=reward_shaping_conf["time_window"])

    def get_agent2policy(self) -> Dict[Any, str]:
        return {agent: f"{algorithm}_{agent}.policy" for agent in self.env.agent_idx_list}

    def get_policy_creator(self) -> Dict[str, Callable[[], AbsPolicy]]:
        if algorithm == "quantum_dqn":
            policy_creator = {
                f"{algorithm}_{i}.policy": partial(creators[algorithm]["policy"], state_dim, action_num, f"{algorithm}_{i}.policy")
                for i in range(num_agents)
            }
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        return policy_creator

    def get_trainer_creator(self) -> Dict[str, Callable[[], AbsTrainer]]:
        if algorithm == "quantum_dqn":
            trainer_creator = {
                f"{algorithm}_{i}": partial(creators[algorithm]["trainer"], f"{algorithm}_{i}") for i in range(num_agents)
            }
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        return trainer_creator
