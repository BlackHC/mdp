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
import typing

from blackhc import mdp
from blackhc.mdp.dsl import context


class DslSyntaxError(SyntaxError):
    pass


class TransitionInfo(object):
    """Whether an expression has a state, action or outcome set."""

    def __init__(self, has_action=False, has_state=False, has_outcome=False):
        # Whether there is an action column
        self.has_action = has_action
        # Whether there is a state column
        self.has_state = has_state
        # Whether there is an outcome column
        self.has_outcome = has_outcome

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.__dict__)

    @property
    def fully_specified(self):
        """Whether all three attributes have been provided and transitions are thus fully specified."""
        return self.has_state and self.has_action and self.has_outcome


class Node(object):
    def __init__(self):
        self._transition_info = None

    @property
    def transition_info(self) -> TransitionInfo:
        if not self._transition_info:
            self._transition_info = self.apply(TransitionInfoVisitor())
        return self._transition_info

    def apply(self, visitor: 'NodeVisitor') -> typing.Any:
        return visitor.visit_atom(self)
    
    def __or__(self, other):
        return Alternatives([self]) | other

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.__dict__)


class SupportConjunction(object):
    def __and__(self: Node, right):
        return Conjunction(self, right)


class SupportMapping(object):
    def __gt__(self: Node, outcome):
        return Mapping(self, outcome)


class Transition(object):
    """Immutable transition call (by gentleman's agreement)."""

    def __init__(self, state=None, action=None, outcome=None):
        self.state = state
        self.action = action
        self.outcome = outcome

    def replace(self, state=None, action=None, outcome=None):
        def not_already_set(new, current):
            return not new or not current
        # TODO: replace with value errors!
        assert not_already_set(state, self.state), (state, self.state)
        assert not_already_set(action, self.action), (action, self.action)
        assert not_already_set(outcome, self.outcome), (outcome, self.outcome)

        state = state or self.state
        action = action or self.action
        outcome = outcome or self.outcome
        return Transition(state, action, outcome)


class Alternatives(Node, SupportConjunction, SupportMapping):
    def __init__(self, alternatives: typing.List[Node]):
        super().__init__()

        self.alternatives = alternatives

    def apply(self, visitor: 'NodeVisitor'):
        return visitor.visit_alternatives(self)

    def __or__(self, other: Node):
        if isinstance(other, Alternatives):
            # Merging alternatives into a single one.
            return Alternatives(self.alternatives + other.alternatives)
        else:
            return Alternatives(self.alternatives + [other])

    def __mul__(self, prob):
        # noinspection PyUnresolvedReferences
        return Alternatives([alternative * prob for alternative in self.alternatives])
    

class Reward(Node):
    def __init__(self, reward: mdp.Reward):
        super().__init__()

        self.reward = reward

    def __mul__(self, other):
        return Reward(mdp.Reward(self.reward.outcome, self.reward.weight * other))

    def apply(self, visitor: 'NodeVisitor'):
        return visitor.visit_reward(self)


class Action(Node, SupportConjunction, SupportMapping):
    def __init__(self, action):
        super().__init__()

        self.action = action

    def apply(self, visitor: 'NodeVisitor'):
        return visitor.visit_action(self)


class State(Node, SupportConjunction, SupportMapping):
    def __init__(self, state):
        super().__init__()

        self.state = state

    def __mul__(self, other):
        return WeightedState(mdp.NextState(self.state, other))

    def __gt__(self, other):
        return Mapping(self, other)

    def apply(self, visitor: 'NodeVisitor'):
        return visitor.visit_state(self)


class WeightedState(Node):
    def __init__(self, next_state: mdp.NextState):
        super().__init__()

        self.next_state = next_state

    def __mul__(self, other):
        return WeightedState(mdp.NextState(self.next_state.outcome, self.next_state.weight * other))

    def apply(self, visitor: 'NodeVisitor'):
        return visitor.visit_weighted_state(self)


class Conjunction(Node, SupportMapping):
    def __init__(self, left: Node, right: Node):
        super().__init__()

        left.apply(TriggerTypeVerifier())
        # right can be either a mapping (so have an outcome) or be a trigger
        if not right.transition_info.has_outcome:
            # right must be a trigger
            right.apply(TriggerTypeVerifier())
        # else: Mapping has already verified itself.

        self.left = left
        self.right = right

        if self.transition_info.fully_specified:
            compile_transitions(self)

    def apply(self, visitor: 'NodeVisitor'):
        return visitor.visit_conjunction(self)


class Mapping(Node):
    def __init__(self, trigger: Node, outcome: Node):
        super().__init__()

        trigger.apply(TriggerTypeVerifier())
        outcome.apply(OutcomeTypeVerifier())

        self.trigger = trigger
        self.outcome = outcome

        if self.transition_info.fully_specified:
            compile_transitions(self)

    def apply(self, visitor: 'NodeVisitor'):
        return visitor.visit_mapping(self)


def compile_transitions(node):
    transitions = node.apply(TransitionVisitor([Transition()]))
    for transition in transitions:
        if transition.state.terminal_state:
            raise DslSyntaxError('Attempted to specify transition %s for terminal state!' % transition)
        context.mdp_spec.transition(transition.state, transition.action, transition.outcome)


class NodeVisitor(object):
    def visit_atom(self, node: Node):
        raise AssertionError('This should be unreachable!')

    def visit_alternatives(self, node: Alternatives):
        return self.visit_atom(node)

    def visit_reward(self, node: Reward):
        return self.visit_atom(node)

    def visit_action(self, node: Action):
        return self.visit_atom(node)

    def visit_state(self, node: State):
        return self.visit_atom(node)

    def visit_weighted_state(self, node: WeightedState):
        return self.visit_atom(node)

    def visit_conjunction(self, node: Conjunction):
        return self.visit_atom(node)

    def visit_mapping(self, node: Mapping):
        return self.visit_atom(node)


class OutcomeTypeVerifier(NodeVisitor):
    """Verifies that all nodes are valid as outcomes."""

    def fail(self, node):
        raise DslSyntaxError('%s not a valid outcome of a transition (either Reward or State)!' % node)

    def visit_weighted_state(self, node: WeightedState):
        pass

    def visit_alternatives(self, node: Alternatives):
        for alternative in node.alternatives:
            alternative.apply(self)

    def visit_mapping(self, node: Mapping):
        # Cannot be valid because a mapping's left side cannot be an outcome.
        self.fail(node)

    def visit_action(self, node: Action):
        self.fail(node)

    def visit_conjunction(self, node: Conjunction):
        # A conjunction cannot be an outcome.
        self.fail(node)

    def visit_reward(self, node: Reward):
        pass

    def visit_state(self, node: State):
        pass


class TriggerTypeVerifier(NodeVisitor):
    """Verifies that all types are valid as triggers."""

    def fail(self, node):
        raise DslSyntaxError('%s not a valid trigger for a transition (either a State or an Action)!' % node)

    def visit_alternatives(self, node: Alternatives):
        for alternative in node.alternatives:
            alternative.apply(self)
        if not node.transition_info.has_action and not node.transition_info.has_state:
            raise DslSyntaxError("%s contains non-homogeneous alternatives!" % node)

    def visit_reward(self, node: Reward):
        pass

    def visit_weighted_state(self, node: WeightedState):
        self.fail(node)

    def visit_state(self, node: State):
        pass

    def visit_mapping(self, node: Mapping):
        self.fail(node)

    def visit_conjunction(self, node: Conjunction):
        node.left.apply(self)
        node.right.apply(self)

    def visit_action(self, node: Action):
        pass


class TransitionInfoVisitor(NodeVisitor):
    def visit_weighted_state(self, node: WeightedState):
        return TransitionInfo()

    def visit_alternatives(self, node: Alternatives):
        has_action = True
        has_state = True
        has_outcome = True
        for alternative in node.alternatives:
            has_state = has_state and alternative.transition_info.has_state
            has_action = has_action and alternative.transition_info.has_action
            has_outcome = has_outcome and alternative.transition_info.has_outcome
        return TransitionInfo(has_state=has_state, has_action=has_action, has_outcome=has_outcome)

    def visit_action(self, node: Action):
        return TransitionInfo(has_action=True)

    def visit_state(self, node: State):
        return TransitionInfo(has_state=True)

    def visit_reward(self, node: Reward):
        return TransitionInfo()

    def visit_conjunction(self, node: Conjunction):
        left_transition_info = node.left.transition_info
        right_transition_info = node.right.transition_info

        if left_transition_info.has_state and right_transition_info.has_state:
            raise DslSyntaxError('%s and %s both have state information!' % (node.left, node.right))
        if left_transition_info.has_action and right_transition_info.has_action:
            raise DslSyntaxError('%s and %s both have action information!' % (node.left, node.right))

        return TransitionInfo(has_action=left_transition_info.has_action or right_transition_info.has_action,
                              has_state=left_transition_info.has_state or right_transition_info.has_state,
                              has_outcome=right_transition_info.has_outcome)

    def visit_mapping(self, node: Mapping):
        return TransitionInfo(has_action=node.trigger.transition_info.has_action,
                              has_state=node.trigger.transition_info.has_state,
                              has_outcome=True)


class TransitionVisitor(NodeVisitor):
    """Returns a generator yielding all transitions.

    Assumes that the root node's transition info is fully specified."""
    def __init__(self, iterator: typing.Iterable[Transition]):
        self.iterator = iterator

    def visit_action(self, node: Action):
        for transition in self.iterator:
            yield transition.replace(action=node.action)

    def visit_state(self, node: State):
        for transition in self.iterator:
            yield transition.replace(state=node.state)

    def visit_alternatives(self, node: Alternatives):
        transitions = list(self.iterator)
        for alternative in node.alternatives:
            yield from alternative.apply(TransitionVisitor(transitions))

    def visit_mapping(self, node: Mapping):
        return node.outcome.apply(TransitionOutcomeVisitor(node.trigger.apply(self)))

    def visit_conjunction(self, node: Conjunction):
        return node.right.apply(TransitionVisitor(node.left.apply(self)))


class TransitionOutcomeVisitor(NodeVisitor):
    def __init__(self, iterator: typing.Iterable[Transition]):
        self.iterator = iterator

    def visit_alternatives(self, node: Alternatives):
        transitions = list(self.iterator)
        for alternative in node.alternatives:
            yield from alternative.apply(TransitionOutcomeVisitor(transitions))

    def visit_reward(self, node: Reward):
        for transition in self.iterator:
            yield transition.replace(outcome=node.reward)

    def visit_weighted_state(self, node: WeightedState):
        for transition in self.iterator:
            yield transition.replace(outcome=node.next_state)

    def visit_state(self, node: State):
        for transition in self.iterator:
            yield transition.replace(outcome=mdp.NextState(node.state))
