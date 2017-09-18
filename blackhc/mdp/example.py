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
from blackhc.mdp import dsl


# noinspection PyStatementEffect
def _one_round_dmdp():
    with dsl.new() as mdp:
        start = dsl.state()
        end = dsl.terminal_state()

        action_0 = dsl.action()
        action_1 = dsl.action()

        start & (action_0 | action_1) > end
        start & action_1 > dsl.reward(1.)

        return mdp.validate()


# noinspection PyStatementEffect
def _two_round_dmdp():
    with dsl.new() as mdp:
        start = dsl.state()
        better = dsl.state()
        worse = dsl.state()
        end = dsl.terminal_state()

        action_0 = dsl.action()
        action_1 = dsl.action()

        start & action_0 > better
        better & action_1 > dsl.reward(3)

        start & action_1 > worse
        worse & action_0 > dsl.reward(1)
        worse & action_1 > dsl.reward(2)

        (better | worse) & (action_0 | action_1) > end

        return mdp.validate()


# noinspection PyStatementEffect
def _one_round_nmdp():
    with dsl.new() as mdp:
        start = dsl.state()
        end = dsl.terminal_state()

        action_0 = dsl.action()
        action_1 = dsl.action()

        start & action_0 > dsl.reward(0) | dsl.reward(5)
        start & action_1 > dsl.reward(1) | dsl.reward(3)

        start & (action_0 | action_1) > end

        return mdp.validate()


# noinspection PyStatementEffect
def _two_round_nmdp():
    with dsl.new() as mdp:
        start = dsl.state()
        a = dsl.state()
        b = dsl.state()
        end = dsl.terminal_state()

        action_0 = dsl.action()
        action_1 = dsl.action()

        start & action_0 > a
        a & action_0 > dsl.reward(-1) | dsl.reward(1)
        a & action_1 > dsl.reward(0) * 2 | dsl.reward(9)

        start & action_1 > b
        b & action_0 > dsl.reward(0) | dsl.reward(2)
        b & action_1 > dsl.reward(2) | dsl.reward(3)

        (a | b) & (action_0 | action_1) > end

        return mdp.validate()


# noinspection PyStatementEffect,PyPep8Naming
def  _multi_round_nmdp():
    with dsl.new() as mdp:
        start = dsl.state()
        end = dsl.terminal_state()

        start & dsl.action() > dsl.reward(5) | start | end * 2
        start & dsl.action() > dsl.reward(3) | start * 2 | end

        dsl.discount(0.5)

        return mdp.validate()


ONE_ROUND_DMDP = _one_round_dmdp()
TWO_ROUND_DMDP = _two_round_dmdp()

ONE_ROUND_NMDP = _one_round_nmdp()
TWO_ROUND_NMDP = _two_round_nmdp()

MULTI_ROUND_NDMP = _multi_round_nmdp()