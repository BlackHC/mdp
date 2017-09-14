from collections import defaultdict, OrderedDict

import gym
import gym.spaces
import networkx as nx
import numpy as np
import typing


class State(object):
    def __init__(self, name, index, terminal_state=False):
        self.name = name
        self.index = index
        self.terminal_state = terminal_state

    def __repr__(self):
        return 'State(%s, %s, %s)' % (self.name, self.index, self.terminal_state)


class Action(object):
    def __init__(self, name, index):
        self.name = name
        self.index = index

    def __repr__(self):
        return 'Action(%s, %s)' % (self.name, self.index)


class Outcome(object):
    """An outcome can be either a reward or a next state.

    For a given (state, action) transition all potential outcomes
    are weighted according to their `weight` and normalized.
    """

    def __init__(self, outcome, weight):
        self.weight = weight
        self.outcome = outcome

    @staticmethod
    def get_choices(outcomes: typing.Iterable['Outcome']):
        """Normalize outcomes and deduplicate into a Dict[outcome, probability]."""

        # Deduplicate elements
        deduped_outcomes = defaultdict(float)
        total_weight = 0.
        for outcome in outcomes:
            deduped_outcomes[outcome.outcome] += outcome.weight
            total_weight += outcome.weight

        return {outcome: weight / total_weight for outcome, weight in deduped_outcomes.items()}


class Reward(Outcome):
    def __init__(self, value, weight=1.0):
        super().__init__(value, weight)

    def __repr__(self):
        return 'Reward(%s, %s)' % (self.outcome, self.weight)

    @staticmethod
    def get_choices(rewards: typing.Iterable['Reward']):
        return Outcome.get_choices(rewards) or {0.: 1.}


class NextState(Outcome):
    def __init__(self, state, weight=1.0):
        super().__init__(state, weight)

    def __repr__(self):
        return 'NextState(%s, %s)' % (self.outcome, self.weight)

    @staticmethod
    def get_choices(next_states: typing.Iterable['NextState']):
        return Outcome.get_choices(next_states)


class MDPSpec(object):
    def __init__(self):
        self._states = {}
        self._actions = {}
        self.states = []
        self.actions = []
        self.state_outcomes: typing.Dict[tuple, typing.List[NextState]] = defaultdict(list)
        self.reward_outcomes: typing.Dict[tuple, typing.List[Reward]] = defaultdict(list)
        self.discount = 1.0

    def state(self, name=None, terminal_state=False):
        if not name:
            if not terminal_state:
                name = 'S%s' % self.num_states
            else:
                name = 'T%s' % self.num_states

        if name not in self.states:
            new_state = State(name, self.num_states, terminal_state=terminal_state)
            self._states[name] = new_state
            self.states.append(new_state)
        return self._states[name]

    def action(self, name):
        if not name:
            name = 'A%s' % self.num_actions

        if name not in self.actions:
            new_action = Action(name, self.num_actions)
            self._actions[name] = new_action
            self.actions.append(new_action)
        return self._actions[name]

    def transition(self, state: State, action: Action, outcome: Outcome):
        """Specify either a next state or a reward as `outcome` for a transition."""

        if isinstance(outcome, NextState):
            self.state_outcomes[state, action].append(outcome)
        elif isinstance(outcome, Reward):
            self.reward_outcomes[state, action].append(outcome)
        else:
            raise NotImplementedError()

    @property
    def num_states(self):
        return len(self._states)

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def is_deterministic(self):
        for state in self.states:
            for action in self.actions:
                if len(self.reward_outcomes[state, action]) > 1:
                    return False
                if len(self.state_outcomes[state, action]) > 1:
                    return False
        return True

    def __repr__(self):
        return 'Mdp(states=%s, actions=%s, state_outcomes=%s, reward_outcomes=%s)' % (
            self.states, self.actions, dict(self.state_outcomes), dict(self.reward_outcomes))

    def to_graph(self, highlight_state: State = None, highlight_action: Action = None,
                 highlight_next_state: State = None):
        transitions = Transitions(self)

        graph = nx.MultiDiGraph()
        for state in self.states:
            graph.add_node(state, label=state.name)
            attributes = graph.node[state]
            if state.terminal_state:
                attributes['shape'] = 'doublecircle'
            if state == highlight_state:
                attributes['fillcolor'] = 'yellow'
                attributes['style'] = 'filled'
            if state == highlight_next_state:
                attributes['fillcolor'] = 'red'
                attributes['style'] = 'filled'

        for state in self.states:
            if not state.terminal_state:
                for action in self.actions:
                    reward_probs = transitions.rewards[state, action].items()
                    expected_reward = sum(reward * prob for reward, prob in reward_probs)
                    stddev_reward = (sum(
                        reward * reward * prob for reward, prob in
                        reward_probs) - expected_reward * expected_reward) ** 0.5

                    action_label = '%s %+.2f' % (action.name, expected_reward)
                    if len(reward_probs) > 1:
                        action_label += ' (%.2f)' % stddev_reward

                    next_states = transitions.next_states[state, action].items()
                    if len(next_states) > 1:
                        transition = (state, action)

                        graph.add_node(transition, shape='point')
                        graph.add_edge(state, transition,
                                       label=action_label)

                        for next_state, prob in next_states:
                            if not prob:
                                continue
                            graph.add_edge(transition, next_state, label='%3.2f%%' % (prob * 100))

                        if state == highlight_state and action == highlight_action:
                            attributes = graph.node[transition]
                            attributes['style'] = 'bold'

                            attributes = graph.edge[state][transition][0]
                            attributes['style'] = 'bold'
                            attributes['color'] = 'green'
                            if highlight_next_state:
                                # Could also check that highlight_next_state is really a next state.
                                attributes = graph.edge[transition][highlight_next_state][0]
                                attributes['style'] = 'bold'
                                attributes['color'] = 'red'
                    else:
                        next_state, _ = list(next_states)[0]
                        graph.add_edge(state, next_state, key=action,
                                       label=action_label)
                        if state == highlight_state and action == highlight_action:
                            attributes = graph.edge[state][next_state][action]
                            attributes['style'] = 'bold'
                            attributes['color'] = 'red'

        return graph

    def to_env(self):
        return MDPEnv(self)

    def validate(self):
        # For now, just validate by trying to compute the transitions.
        # It will raise errors if anything is wrong.
        Transitions(self)
        return self


class Transitions(object):
    """Container for transition probabilities."""

    def __init__(self, mdp: MDPSpec):
        self.next_states = {}
        self.rewards = {}
        for state in mdp.states:
            for action in mdp.actions:
                next_states = NextState.get_choices(mdp.state_outcomes[state, action])
                if not state.terminal_state and not next_states:
                    raise ValueError('No next states specified for non-terminal (%s, %s)!' % (state, action))
                if state.terminal_state and next_states:
                    raise ValueError('Next states %s specified for terminal (%s, %s)!' % (next_states, state, action))
                self.next_states[state, action] = next_states

                rewards = mdp.reward_outcomes[state, action]
                if state.terminal_state and rewards:
                    raise ValueError('Rewards %s specified for terminal (%s, %s)!' % (next_states, state, action))
                self.rewards[state, action] = Reward.get_choices(rewards)

    def __repr__(self):
        return 'Transitions(%s)' % self.__dict__


class MDPEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'png']}

    def __init__(self, mdp: MDPSpec, start_state: State = None):
        self.render_widget = None

        self.mdp = mdp
        self.transitions = Transitions(mdp)

        self._previous_state: State = None
        self._previous_action: Action = None
        self._state: State = None
        self._is_done = True
        self.observation_space = gym.spaces.Discrete(self.mdp.num_states)
        self.action_space = gym.spaces.Discrete(self.mdp.num_actions)
        self.start_state = start_state or list(self.mdp.states)[0]

    def _reset(self):
        self._previous_state = None
        self._previous_action = None
        self._state = self.start_state
        self._is_done = self._state.terminal_state
        return self._state.index

    def _step(self, action_index):
        action = self.mdp.actions[action_index]
        self._previous_state = self._state
        self._previous_action = action

        if not self._is_done:
            reward_probs = self.transitions.rewards[self._state, action]
            reward = np.random.choice(list(reward_probs.keys()), p=list(reward_probs.values()))

            next_state_probs = self.transitions.next_states[self._state, action]
            self._state = np.random.choice(list(next_state_probs.keys()), p=list(next_state_probs.values()))
            self._is_done = self._state.terminal_state
        else:
            reward = 0

        return self._state.index, reward, self._is_done, None

    def to_graph(self):
        graph = self.mdp.to_graph(highlight_state=self._previous_state, highlight_action=self._previous_action,
                                  highlight_next_state=self._state)
        return graph

    def _render(self, mode='human', close=False):
        if close:
            if self.render_widget:
                self.render_widget.close()
            return

        png_data = graph_to_png(self.to_graph())

        if mode == 'human':
            # TODO: use OpenAI's SimpleImageViewer wrapper when not running in IPython.
            if not self.render_widget:
                from IPython.display import display
                import ipywidgets as widgets

                self.render_widget = widgets.Image()
                display(self.render_widget)

            self.render_widget.value = png_data
        elif mode == 'rgb_array':
            from matplotlib import pyplot
            import io
            return pyplot.imread(io.BytesIO(png_data))
        elif mode == 'png':
            return png_data


def graph_to_png(graph):
    pydot_graph = nx.nx_pydot.to_pydot(graph)
    return pydot_graph.create_png()


def display_mdp(mdp):
    from IPython.display import display, Image
    display(Image(graph_to_png(mdp.to_graph())))
