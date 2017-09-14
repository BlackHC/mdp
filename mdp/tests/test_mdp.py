import mdp
from mdp import dsl


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
