import pytest

import mdp.dsl


# noinspection PyStatementEffect,PyPep8Naming
def test_coverage():
    with mdp.dsl.new() as new_mdp:
        stateA = mdp.dsl.state()
        stateB = mdp.dsl.state()
        actionA = mdp.dsl.action()
        actionB = mdp.dsl.action()

        stateA & actionA > stateA
        stateA & actionB > stateB
        stateB & (actionA | actionB) > stateB

        new_mdp.to_env()
        new_mdp.to_graph()
        return new_mdp.validate()


# noinspection PyStatementEffect
def test_weighted_next_states():
    with mdp.dsl.new() as new_mdp:
        state = mdp.dsl.state()
        action = mdp.dsl.action()

        state & action > state * 0.5
        state & action > state * 2 | state * 5

        new_mdp.validate()


# noinspection PyStatementEffect
def test_weighted_rewards():
    with mdp.dsl.new() as new_mdp:
        state = mdp.dsl.state()
        action = mdp.dsl.action()

        state & action > mdp.dsl.reward(1) * 1
        state & action > mdp.dsl.reward(1) * 1 | mdp.dsl.reward(2) * 3


# noinspection PyStatementEffect,PyPep8Naming
def test_alternatives():
    with mdp.dsl.new():
        stateA = mdp.dsl.state()
        stateB = mdp.dsl.state()
        actionA = mdp.dsl.action()
        actionB = mdp.dsl.action()

        (stateA | stateB) & (actionA | actionB) > (stateA | stateB)

        mdp.dsl.to_env()


# noinspection PyStatementEffect,PyPep8Naming
def test_alternatives2():
    with mdp.dsl.new():
        stateA = mdp.dsl.state()
        stateB = mdp.dsl.state()
        actionA = mdp.dsl.action()
        actionB = mdp.dsl.action()

        (stateA | stateB) & (actionA | actionB > stateA | stateB)

        mdp.dsl.to_env()


# noinspection PyStatementEffect,PyPep8Naming
def test_alternatives3():
    with mdp.dsl.new():
        stateA = mdp.dsl.state()
        stateB = mdp.dsl.state()
        actionA = mdp.dsl.action()
        actionB = mdp.dsl.action()

        (stateA | stateB) & ((actionA > stateA) | (actionB > stateB))

        mdp.dsl.to_env()


# noinspection PyStatementEffect,PyPep8Naming
def test_coverage_nmrp():
    with mdp.dsl.new():
        stateA = mdp.dsl.state()
        stateB = mdp.dsl.state()
        actionA = mdp.dsl.action()
        actionB = mdp.dsl.action()

        stateA & actionA > stateA
        stateA & actionB > stateB
        stateB & (actionA | actionB) > stateB

        mdp.dsl.to_env()


# noinspection PyStatementEffect,PyPep8Naming
def test_multi_states_fail():
    with pytest.raises(mdp.dsl.SyntaxError):
        with mdp.dsl.new():
            stateA = mdp.dsl.state()
            stateB = mdp.dsl.state()

            stateA & stateB


# noinspection PyStatementEffect,PyPep8Naming
def test_multi_actions_fail():
    with pytest.raises(mdp.dsl.SyntaxError):
        with mdp.dsl.new():
            actionA = mdp.dsl.action()
            actionB = mdp.dsl.action()

            actionA & actionB


# noinspection PyStatementEffect,PyPep8Naming
def test_alternative_mismatch_final_fail():
    with pytest.raises(mdp.dsl.SyntaxError):
        with mdp.dsl.new():
            stateA = mdp.dsl.state()
            actionB = mdp.dsl.action()

            stateA | actionB > stateA


# noinspection PyStatementEffect,PyPep8Naming
def test_mapping_alternative_mismatch_fail():
    with pytest.raises(mdp.dsl.SyntaxError):
        with mdp.dsl.new():
            stateA = mdp.dsl.state()
            actionB = mdp.dsl.action()

            stateA > stateA | actionB


def test_missing_terminal_state_fail():
    with pytest.raises(ValueError):
        with mdp.dsl.new() as new_mdp:
            mdp.dsl.state()
            mdp.dsl.action()

            new_mdp.validate()
