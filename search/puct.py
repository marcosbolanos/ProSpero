from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Generic, Hashable, Iterable, Protocol, TypeVar


ActionT = TypeVar("ActionT", bound=Hashable)
StateT = TypeVar("StateT")


@dataclass(frozen=True)
class ActionPrior(Generic[ActionT]):
    action: ActionT
    prior: float
    immediate_value: float = 0.0


@dataclass(frozen=True)
class SearchResult(Generic[StateT]):
    state: StateT
    value: float
    visits: int


class PUCTEvaluator(Protocol[StateT, ActionT]):
    def is_terminal(self, state: StateT) -> bool:
        ...

    def terminal_value(self, state: StateT) -> float:
        ...

    def actions(self, state: StateT) -> Iterable[ActionPrior[ActionT]]:
        ...

    def transition(self, state: StateT, action: ActionT, immediate_value: float) -> StateT:
        ...

    def rollout_value(self, state: StateT) -> float:
        ...


@dataclass
class PUCTConfig:
    simulations: int = 64
    c_puct: float = 1.5

    def __post_init__(self):
        if self.simulations < 1:
            raise ValueError("simulations must be >= 1")
        if self.c_puct < 0:
            raise ValueError("c_puct must be non-negative")


@dataclass
class _Edge(Generic[ActionT, StateT]):
    action: ActionT
    prior: float
    immediate_value: float
    child: "_Node[ActionT, StateT] | None" = None
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / float(self.visits)


@dataclass
class _Node(Generic[ActionT, StateT]):
    state: StateT
    expanded: bool = False
    visits: int = 0
    edges: dict[ActionT, _Edge[ActionT, StateT]] = field(default_factory=dict)

    def expand(self, evaluator: PUCTEvaluator[StateT, ActionT]):
        if self.expanded:
            return
        action_priors = list(evaluator.actions(self.state))
        if not action_priors and not evaluator.is_terminal(self.state):
            raise ValueError("Non-terminal PUCT state had no legal actions.")
        prior_sum = sum(max(0.0, float(item.prior)) for item in action_priors)
        if prior_sum <= 0.0 and action_priors:
            uniform = 1.0 / len(action_priors)
            self.edges = {
                item.action: _Edge(
                    action=item.action,
                    prior=uniform,
                    immediate_value=float(item.immediate_value),
                )
                for item in action_priors
            }
        else:
            self.edges = {
                item.action: _Edge(
                    action=item.action,
                    prior=max(0.0, float(item.prior)) / prior_sum,
                    immediate_value=float(item.immediate_value),
                )
                for item in action_priors
            }
        self.expanded = True


class PUCTSearch(Generic[StateT, ActionT]):
    """Small auditable PUCT implementation for shallow protein decoding trees."""

    def __init__(self, evaluator: PUCTEvaluator[StateT, ActionT], config: PUCTConfig | None = None):
        self.evaluator = evaluator
        self.config = config or PUCTConfig()
        self.root: _Node[ActionT, StateT] | None = None
        self.terminal_results: dict[StateT, SearchResult[StateT]] = {}

    def run(self, initial_state: StateT) -> list[SearchResult[StateT]]:
        self.root = _Node(initial_state)
        self.terminal_results = {}
        for _ in range(self.config.simulations):
            self._simulate(self.root)
        if self.evaluator.is_terminal(initial_state):
            value = self.evaluator.terminal_value(initial_state)
            self.terminal_results[initial_state] = SearchResult(initial_state, value, self.root.visits)
        return self.results()

    def results(self) -> list[SearchResult[StateT]]:
        return sorted(
            self.terminal_results.values(),
            key=lambda item: (item.value, item.visits),
            reverse=True,
        )

    def _simulate(self, root: _Node[ActionT, StateT]) -> float:
        node = root
        path: list[_Edge[ActionT, StateT]] = []

        while True:
            node.visits += 1
            if self.evaluator.is_terminal(node.state):
                value = self.evaluator.terminal_value(node.state)
                self._record_terminal(node.state, value, node.visits)
                break

            if not node.expanded:
                node.expand(self.evaluator)
                value = self.evaluator.rollout_value(node.state)
                break

            edge = self._select_edge(node)
            path.append(edge)
            if edge.child is None:
                child_state = self.evaluator.transition(node.state, edge.action, edge.immediate_value)
                edge.child = _Node(child_state)
            node = edge.child

        for edge in path:
            edge.visits += 1
            edge.value_sum += value
        return value

    def _select_edge(self, node: _Node[ActionT, StateT]) -> _Edge[ActionT, StateT]:
        if not node.edges:
            raise ValueError("Cannot select from a node with no edges.")
        parent_visits = max(1, node.visits)
        exploration_scale = math.sqrt(parent_visits)

        def score(edge: _Edge[ActionT, StateT]) -> float:
            exploration = self.config.c_puct * edge.prior * exploration_scale / (1.0 + edge.visits)
            return edge.q_value + exploration

        return max(node.edges.values(), key=score)

    def _record_terminal(self, state: StateT, value: float, visits: int):
        previous = self.terminal_results.get(state)
        if previous is None or value > previous.value or visits > previous.visits:
            self.terminal_results[state] = SearchResult(state=state, value=float(value), visits=int(visits))

