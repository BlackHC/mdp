import contextlib
import mdp
from mdp.dsl import ast
from mdp.dsl import context as dsl_context


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
