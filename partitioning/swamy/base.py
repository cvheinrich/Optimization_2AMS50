from typing import Dict, List, Tuple
from partitioning.base import DistrictPartitioner


class BaseSwamyPartitioner(DistrictPartitioner):
    SLACK_FIXED = "fixed"
    SLACK_VARIABLE = "var"
    SLACK_TYPES = [SLACK_FIXED, SLACK_VARIABLE]
    SLACK_DEFAULT = SLACK_VARIABLE

    ALPHA_DEFAULT = 0.5
    SLACK_VALUE_DEFAULT = 0.025
    MAX_ITER_DEFAULT = 500

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
        max_iter: int = MAX_ITER_DEFAULT,
    ):
        """
        Initialize base partitioner
        """
        super().__init__(state, K, G, P, D)
        self.alpha = alpha
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        self.slack_type = slack_type
        self.slack_value = slack_value
        self.max_iter = max_iter
        # Calculate normalization constant to attempt to make alpha more sensible
        self.C = self.total_population / (
            sum(self.distances[i][j] for i in self.edges for j in self.edges[i])
        )
        print(f"C = {self.C}")

    def from_files(
        state: str,
        alpha: float = ALPHA_DEFAULT,
        slack_type: str = SLACK_DEFAULT,
        slack_value: float = SLACK_VALUE_DEFAULT,
        max_iter: int = MAX_ITER_DEFAULT,
    ) -> "BaseSwamyPartitioner":
        """
        Initialize base partitioner from files
        """
        return BaseSwamyPartitioner(
            state,
            *BaseSwamyPartitioner._read_files(state),
            alpha,
            slack_type,
            slack_value,
            max_iter,
        )

    def prepare_graph(
        self, remove_large_nodes: bool = True
    ) -> Tuple[Dict[int, List[int]], List[int], List[List[int]], List[int]]:
        """
        @param remove_large_nodes: If True, remove large nodes from the graph
        @return: Tuple of (G', P', D', large_nodes), where G',P',D' are created by removing large nodes,
                 if `remove_large_nodes` is True, otherwise return the original G,P,D
        """
        limit = self.avg_population + self.slack_value

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

    def _is_balanced(
        self, prev_part_nodes: List[int], new_part_nodes: List[int], P: Dict[int, int]
    ) -> bool:
        return (
            abs(sum(P[i] for i in prev_part_nodes) - self.avg_population) < self.slack_value
            and abs(sum(P[i] for i in new_part_nodes) - self.avg_population) < self.slack_value
        )

    def _is_connected(self, part_nodes: List[int], G: Dict[int, List[int]]) -> bool:
        """
        Check if the partition is connected
        """
        visited = {part_nodes[0]}
        stack = [part_nodes[0]]
        while stack:
            node = stack.pop()
            for neighbor in G[node]:
                if neighbor in part_nodes and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        return len(visited) == len(part_nodes)

    def _local_optimize(
        self,
        G: Dict[int, List[int]],
        P: Dict[int, int],
        D: Dict[Tuple[int, int], int],
        partitions: Dict[int, List[int]],
    ) -> Dict[int, List[int]]:
        """
        Local optimization heuristic
        """
        node_partitions = {node: k for k, partition in partitions.items() for node in partition}

        def cost_increase(node: int, part_B: int) -> int:
            part_A = node_partitions[node]

            dists_in_new = sum(D[node, i] for i in partitions[part_B])
            dists_in_prev = sum(D[node, i] for i in partitions[part_A] if i != node)

            pop_A = sum(P[i] for i in partitions[part_A])
            pop_B = sum(P[i] for i in partitions[part_B])

            return (1 - self.alpha) * (dists_in_new - dists_in_prev) + self.C * self.alpha * (
                abs(pop_A - P[node] - self.avg_population)
                + abs(pop_B + P[node] - self.avg_population)
                - abs(pop_A - self.avg_population)
                - abs(pop_B - self.avg_population)
            )

        # key: (node, partition), value: (cost increase, number of neighbors in partition)
        border_nodes: Dict[Tuple[int, int], Tuple[int, int]] = {}

        def add_to_border(node, partition):
            if (node, partition) not in border_nodes:
                border_nodes[node, partition] = (cost_increase(node, partition), 1)
            else:
                increase, count = border_nodes[node, partition]
                border_nodes[node, partition] = (increase, count + 1)

        for i in G:
            part_i = node_partitions[i]
            for j in G[i]:
                part_j = node_partitions[j]
                if part_i != part_j:
                    add_to_border(i, part_j)

        skip = 0
        for _ in range(self.max_iter):
            if skip == 0:
                sorted_border_nodes = sorted(border_nodes.items(), key=lambda x: x[1][0])
            i, new_part = sorted_border_nodes[skip][0]
            if sorted_border_nodes[skip][1][0] >= 0:
                break

            prev_part = node_partitions[i]
            partitions[prev_part].remove(i)
            # if not self._is_connected(partitions[prev_part], G) or not self._is_balanced(
            #     partitions[prev_part], partitions[new_part], P
            # ):
            if not self._is_connected(partitions[prev_part], G):
                partitions[prev_part].append(i)
                skip += 1
                continue

            partitions[new_part].append(i)
            node_partitions[i] = new_part
            del border_nodes[i, new_part]
            skip = 0

            for node, part in border_nodes:
                node_part = node_partitions[node]
                if node_part in [prev_part, new_part] or part in [prev_part, new_part]:
                    decrease, count = border_nodes[node, part]
                    border_nodes[node, part] = (cost_increase(node, part), count)

            for j in G[i]:
                part_j = node_partitions[j]
                if new_part != part_j:
                    add_to_border(i, part_j)
                    add_to_border(j, new_part)
                else:
                    decrease, count = border_nodes[j, prev_part]
                    if count == 1:
                        del border_nodes[j, prev_part]
                    else:
                        border_nodes[j, prev_part] = (decrease, count - 1)

        return partitions