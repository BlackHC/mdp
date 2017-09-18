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
"""Linear programming solver for MDPs.

This is a very basic solver.
"""

import numpy as np

from blackhc import mdp


class LinearProgramming(object):
    def __init__(self, mdp_spec: mdp.MDPSpec):
        self.discount = mdp_spec.discount
        self.num_states = mdp_spec.num_states
        self.num_actions = mdp_spec.num_actions
        self.mdp_spec = mdp_spec

        transitions = mdp.Transitions(mdp_spec)
        next_states = np.zeros(shape=(self.num_states, self.num_actions, self.num_states))
        expected_rewards = np.zeros(shape=(self.num_states, self.num_actions))
        for (state, action), choices in transitions.next_states.items():
            for next_state, prob in choices.items():
                next_states[state.index, action.index, next_state.index] = prob
        for state in mdp_spec.states:
            if state.terminal_state:
                next_states[state.index, :, state.index] = 1.

        for (state, action), choices in transitions.rewards.items():
            expected_rewards[state.index, action.index] = sum(value * prob for value, prob in choices.items())

        self.next_states = next_states
        self.expected_rewards = expected_rewards

    def compute_q_table(self, max_iterations=100, all_close=None):
        return _fix_point_iterate(self.expected_rewards.copy(),
                                  lambda q_table: self.q_table_from_v_vector(self.v_vector_from_q_table(q_table)),
                                  max_iterations=max_iterations,
                                  all_close=all_close
                                  )

    def compute_v_vector(self, max_iterations=100, all_close=None):
        return _fix_point_iterate(np.zeros((self.num_states,)),
                                  lambda v_vector: self.v_vector_from_q_table(self.q_table_from_v_vector(v_vector)),
                                  max_iterations=max_iterations,
                                  all_close=all_close)

    # noinspection PyMethodMayBeStatic
    def v_vector_from_q_table(self, q_table):
        v_vector = q_table.max(axis=-1)
        return v_vector

    def q_table_from_v_vector(self, v_vector):
        # TODO: somebody said that q table converges faster than v table
        # Is that true? What if take computation cost into account?
        # TODO: support passing it in as a parameter
        q_table = self.expected_rewards + self.discount * (
            self.next_states * v_vector.reshape((1, 1, self.num_states))).sum(axis=-1)
        return q_table


def _fix_point_iterate(initial, iterate, max_iterations, all_close=None):
    if not all_close:
        all_close = np.allclose

    value = initial
    for _ in range(max_iterations):
        next_value = iterate(value)
        if all_close(value, next_value):
            return next_value
        value = next_value
    raise ValueError('No convergence after %s iterations!\n%s' % (max_iterations, value))
