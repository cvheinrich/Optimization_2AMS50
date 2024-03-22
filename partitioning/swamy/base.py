from typing import Dict, List, Tuple
from partitioning.base import DistrictPartitioner


class BaseSwamyPartitioner(DistrictPartitioner):
    SLACK_FIXED = "fixed"
    SLACK_VARIABLE = "var"
    SLACK_TYPES = [SLACK_FIXED, SLACK_VARIABLE]
    SLACK_DEFAULT = SLACK_FIXED

    ALPHA_DEFAULT = 1.0
    SLACK_VALUE_DEFAULT = 0.025

    def __init__(
        self,
        state: str,
        K: int,
        G: Dict[int, List[int]],
        P: List[int],
        D: List[List[int]],
        alpha: float = ALPHA_DEFAULT,
        slack_type: str = SLACK_DEFAULT,
        slack_value: float = SLACK_VALUE_DEFAULT,
    ):
        """
        Initialize base partitioner
        """
        super().__init__(state, K, G, P, D)
        self.alpha = alpha
        self.slack_type = slack_type
        self.slack_value = slack_value

    def from_files(
        state: str,
        alpha: float = ALPHA_DEFAULT,
        slack_type: str = SLACK_DEFAULT,
        slack_value: float = SLACK_VALUE_DEFAULT,
    ) -> "BaseSwamyPartitioner":
        """
        Initialize base partitioner from files
        """
        return BaseSwamyPartitioner(
            state, *BaseSwamyPartitioner._read_files(state), alpha, slack_type, slack_value
        )

    def isolate_large_nodes(
        self,
    ) -> Tuple[Dict[int, List[int]], List[int], List[List[int]], List[int]]:
        """
        @return: Tuple of (G', P', D', large_nodes), where G',P',D' are created by removing large nodes
        """
        limit = self.avg_population
        if self.slack_type == self.SLACK_FIXED:
            limit += self.slack_value

        large_nodes = [i for i, p in enumerate(self.populations) if p > limit]

        G = {
            i: [j for j in self.edges[i] if j not in large_nodes]
            for i in self.edges
            if i not in large_nodes
        }
        P = [p for i, p in enumerate(self.populations) if i not in large_nodes]
        D = [
            [d for j, d in enumerate(row) if j not in large_nodes]
            for i, row in enumerate(self.distances)
            if i not in large_nodes
        ]

        return G, P, D, large_nodes
