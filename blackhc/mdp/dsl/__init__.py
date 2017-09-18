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
import contextlib

from blackhc import mdp
from blackhc.mdp.dsl import context as dsl_context
from blackhc.mdp.dsl import ast


# noinspection PyShadowingBuiltins
SyntaxError = ast.DslSyntaxError


def state(name=None):
    new_state = dsl_context.mdp_spec.state(name)
    return ast.State(new_state)


def terminal_state(name=None):
    new_state = dsl_context.mdp_spec.state(name, terminal_state=True)
    return ast.State(new_state)


def action(name=None):
    new_action = dsl_context.mdp_spec.action(name)
    return ast.Action(new_action)


def reward(value):
    return ast.Reward(mdp.Reward(value))


def to_env():
    return dsl_context.mdp_spec.to_env()


def to_graph(*args, **kwargs):
    return dsl_context.mdp_spec.to_graph(*args, **kwargs)

def discount(value):
    dsl_context.mdp_spec.discount = value


@contextlib.contextmanager
def new():
    old_context = dsl_context.mdp_spec
    dsl_context.mdp_spec = mdp.MDPSpec()
    yield dsl_context.mdp_spec
    dsl_context.mdp_spec = old_context
