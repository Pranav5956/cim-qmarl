from .quantum_dqn import get_quantum_dqn, get_quantum_dqn_policy

creators = {
    "quantum_dqn": {
        "trainer": get_quantum_dqn,
        "policy": get_quantum_dqn_policy
    }
}
