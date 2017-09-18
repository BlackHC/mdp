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
from blackhc import mdp
from blackhc.mdp import dsl


# noinspection PyStatementEffect
def test_coverage():
    with dsl.new() as new_mdp:  # type: mdp.MDPSpec
        start = dsl.state()
        finish = dsl.terminal_state()

        action = dsl.action()

        start & action > finish | dsl.reward(1)

        assert new_mdp.num_states == 2
        assert new_mdp.num_actions == 1

        new_mdp.to_graph()

        env: mdp.MDPEnv = new_mdp.to_env()

        state = env.reset()
        assert state == 0
        state, reward, is_done, info = env.step(0)
        assert state == 1
        assert reward == 1
        assert is_done

        env.render(mode='rgb_array')
        env.render(mode='png')


def test_without_dsl():
    spec = mdp.MDPSpec()

    start = spec.state('start')
    end = spec.state('end', terminal_state=True)
    action_0 = spec.action()
    action_1 = spec.action()

    spec.transition(start, action_0, mdp.NextState(end))
    spec.transition(start, action_1, mdp.NextState(end))
    spec.transition(start, action_1, mdp.Reward(1))

    spec.validate()