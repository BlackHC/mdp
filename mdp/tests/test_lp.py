import mdp
from mdp import dsl
from mdp import lp
import numpy as np


# noinspection PyStatementEffect
def test_geometric_series():
    with mdp.dsl.new() as new_mdp:
        start = dsl.state()
        action = dsl.action()

        start & action > dsl.reward(1) | start

        dsl.discount(0.5)

        solver = lp.LinearProgramming(new_mdp)

        assert np.allclose(solver.compute_v_vector(), [2.0])
        assert np.allclose(solver.compute_q_table(), [[2.0]])


# noinspection PyStatementEffect
def test_multiple_actions():
    with mdp.dsl.new() as new_mdp:
        start = dsl.state()
        state_a = dsl.state()
        state_b = dsl.state()

        action_a = dsl.action()
        action_b = dsl.action()

        either_action = action_a | action_b

        start & action_a > state_a
        start & action_b > state_b

        state_a & either_action > state_a | dsl.reward(1)
        state_b & either_action > state_b | dsl.reward(2)

        dsl.discount(1/3)

        solver = lp.LinearProgramming(new_mdp)

        assert np.allclose(solver.compute_v_vector(), [1, 1.5, 3])
        assert np.allclose(solver.compute_q_table(), [[0.5, 1], [1.5, 1.5], [3, 3]])
