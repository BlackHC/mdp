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
import pytest
from blackhc.mdp import dsl


# noinspection PyStatementEffect,PyPep8Naming
def test_coverage():
    with dsl.new() as new_mdp:
        stateA = dsl.state()
        stateB = dsl.state()
        actionA = dsl.action()
        actionB = dsl.action()

        stateA & actionA > stateA
        stateA & actionB > stateB
        stateB & (actionA | actionB) > stateB

        new_mdp.to_env()
        new_mdp.to_graph()
        return new_mdp.validate()


# noinspection PyStatementEffect
def test_weighted_next_states():
    with dsl.new() as new_mdp:
        state = dsl.state()
        action = dsl.action()

        state & action > state * 0.5
        state & action > state * 2 | state * 5

        new_mdp.validate()


# noinspection PyStatementEffect
def test_weighted_rewards():
    with dsl.new() as new_mdp:
        state = dsl.state()
        action = dsl.action()

        state & action > dsl.reward(1) * 1
        state & action > dsl.reward(1) * 1 | dsl.reward(2) * 3


# noinspection PyStatementEffect,PyPep8Naming
def test_alternatives():
    with dsl.new():
        stateA = dsl.state()
        stateB = dsl.state()
        actionA = dsl.action()
        actionB = dsl.action()

        (stateA | stateB) & (actionA | actionB) > (stateA | stateB)

        dsl.to_env()


# noinspection PyStatementEffect,PyPep8Naming
def test_alternatives2():
    with dsl.new():
        stateA = dsl.state()
        stateB = dsl.state()
        actionA = dsl.action()
        actionB = dsl.action()

        (stateA | stateB) & (actionA | actionB > stateA | stateB)

        dsl.to_env()


# noinspection PyStatementEffect,PyPep8Naming
def test_alternatives3():
    with dsl.new():
        stateA = dsl.state()
        stateB = dsl.state()
        actionA = dsl.action()
        actionB = dsl.action()

        (stateA | stateB) & ((actionA > stateA) | (actionB > stateB))

        dsl.to_env()


# noinspection PyStatementEffect,PyPep8Naming
def test_coverage_nmrp():
    with dsl.new():
        stateA = dsl.state()
        stateB = dsl.state()
        actionA = dsl.action()
        actionB = dsl.action()

        stateA & actionA > stateA
        stateA & actionB > stateB
        stateB & (actionA | actionB) > stateB

        dsl.to_env()


# noinspection PyStatementEffect,PyPep8Naming
def test_multi_states_fail():
    with pytest.raises(dsl.SyntaxError):
        with dsl.new():
            stateA = dsl.state()
            stateB = dsl.state()

            stateA & stateB


# noinspection PyStatementEffect,PyPep8Naming
def test_multi_actions_fail():
    with pytest.raises(dsl.SyntaxError):
        with dsl.new():
            actionA = dsl.action()
            actionB = dsl.action()

            actionA & actionB


# noinspection PyStatementEffect,PyPep8Naming
def test_alternative_mismatch_final_fail():
    with pytest.raises(dsl.SyntaxError):
        with dsl.new():
            stateA = dsl.state()
            actionB = dsl.action()

            stateA | actionB > stateA


# noinspection PyStatementEffect,PyPep8Naming
def test_mapping_alternative_mismatch_fail():
    with pytest.raises(dsl.SyntaxError):
        with dsl.new():
            stateA = dsl.state()
            actionB = dsl.action()

            stateA > stateA | actionB


def test_missing_terminal_state_fail():
    with pytest.raises(ValueError):
        with dsl.new() as new_mdp:
            dsl.state()
            dsl.action()

            new_mdp.validate()
