from blackhc.mdp import dsl
# Copyright 2017 Andreas Kirsch <blackhc@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest
from blackhc.mdp import lp
from blackhc.mdp import dsl


# noinspection PyStatementEffect
def test_geometric_series():
    with dsl.new() as new_mdp:
        start = dsl.state()
        action = dsl.action()

        start & action > dsl.reward(1) | start

        dsl.discount(0.5)

        solver = lp.LinearProgramming(new_mdp)

        assert np.allclose(solver.compute_v_vector(), [2.0])
        assert np.allclose(solver.compute_q_table(), [[2.0]])


# noinspection PyStatementEffect
def test_multiple_actions():
    with dsl.new() as new_mdp:
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

        dsl.discount(1 / 3)

        solver = lp.LinearProgramming(new_mdp)

        assert np.allclose(solver.compute_v_vector(), [1, 1.5, 3])
        assert np.allclose(solver.compute_q_table(), [[0.5, 1], [1.5, 1.5], [3, 3]])


# noinspection PyStatementEffect
def test_divergence_raises():
    with dsl.new() as new_mdp:
        start = dsl.state()
        action = dsl.action()

        start & action > start | dsl.reward(1)

    solver = lp.LinearProgramming(new_mdp)

    with pytest.raises(ValueError):
        solver.compute_v_vector(max_iterations=10)


def test_terminal_state():
    with dsl.new() as new_mdp:
        dsl.terminal_state()
        dsl.action()

    solver = lp.LinearProgramming(new_mdp)
    assert np.isclose(solver.compute_q_table(), [0])
