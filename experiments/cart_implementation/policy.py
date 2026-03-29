"""UCB-Cost action selection policy — CART's key contribution."""

import math


class UCBCostPolicy:
    """
    Extends UCB1 [Auer et al. 2002] with cost penalty and curiosity terms:

      score(a) = Q̂(a)
               + β √(ln N / n_a)          UCB exploration [Auer 2002]
               + γ √(ln N / (n_a + 1))    curiosity (novel)
               - λ · c(a)                 cost penalty (novel, CART key term)

    where:
      Q̂(a) = running-mean F1 reward for action a
      n_a   = visit count for a; N = total actions taken
      c(a)  = {think: 0.3, retrieve: 0.6, tool: 1.0}

    On stronger models: think rewards Q̂(think) increase naturally,
    shifting the policy toward think without configuration.
    """

    COSTS: dict[str, float] = {"think": 0.3, "retrieve": 0.6, "tool": 1.0}

    def __init__(
        self,
        beta: float = 1.0,
        gamma: float = 0.5,
        lambda_cost: float = 1.0,
    ) -> None:
        self.beta = beta
        self.gamma = gamma
        self.lambda_cost = lambda_cost
        self.Q: dict[str, float] = {}
        self.N: dict[str, int] = {}
        self.total: int = 0

    def select(self) -> str:
        self.total += 1
        best, best_score = None, -float("inf")
        for action, cost in self.COSTS.items():
            n_a = self.N.get(action, 0)
            if n_a == 0:
                return action  # always explore unvisited actions first
            q_a = self.Q.get(action, 0.5)
            ucb = self.beta * math.sqrt(math.log(self.total) / n_a)
            curiosity = self.gamma * math.sqrt(math.log(self.total) / (n_a + 1))
            score = q_a + ucb + curiosity - self.lambda_cost * cost
            if score > best_score:
                best_score, best = score, action
        return best  # type: ignore[return-value]

    def update(self, action: str, reward: float) -> None:
        """Running-mean Q update: Q̂_new = Q̂_old + (r - Q̂_old) / n"""
        self.N[action] = self.N.get(action, 0) + 1
        n = self.N[action]
        q = self.Q.get(action, 0.5)
        self.Q[action] = q + (reward - q) / n
