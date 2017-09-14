from mdp import dsl


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
        first = dsl.state()
        second = dsl.state()
        end = dsl.terminal_state()

        actionA = dsl.action()
        actionB = dsl.action()

        eitherAction = actionA | actionB
        eitherState = first | second

        start & actionA > (first * 0.25 | second * 0.75)
        start & actionB > (first * 0.75 | second * 0.25)

        eitherState & actionA > eitherState
        first & actionB > (first | second * 2)
        second & actionB > (first * 2 | second | end)

        first & eitherAction > dsl.reward(3)
        second & eitherAction > dsl.reward(5)

        dsl.discount(0.9)

        return mdp.validate()

ONE_ROUND_DMDP = _one_round_dmdp()
TWO_ROUND_DMDP = _two_round_dmdp()

ONE_ROUND_NMDP = _one_round_nmdp()
TWO_ROUND_NMDP = _two_round_nmdp()

MULTI_ROUND_NDMP = _multi_round_nmdp()