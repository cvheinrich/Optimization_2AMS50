from typing import Dict, List, Tuple

from partitioning.swamy.base import BaseSwamyPartitioner as BSP
from partitioning.swamy.optimal import OptimalPartitioner


class HeuristicPartitioner(BSP):
    def __init__(
        self,
        state: str,
        K: int,
        G: Dict[int, List[int]],
        P: List[int],
        D: List[List[int]],
        alpha: float = BSP.ALPHA_DEFAULT,
        slack_type: str = BSP.SLACK_DEFAULT,
        slack_value: float = BSP.SLACK_VALUE_DEFAULT,
        max_iter: int = BSP.MAX_ITER_DEFAULT,
    ):
        """
        Initialize Swamy partitioner
        """

        super().__init__(state, K, G, P, D, alpha, slack_type, slack_value, max_iter)

    def from_files(
        state: str,
        alpha: float = BSP.ALPHA_DEFAULT,
        slack_type: str = BSP.SLACK_TYPES,
        slack_value: float = BSP.SLACK_VALUE_DEFAULT,
        max_iter: int = BSP.MAX_ITER_DEFAULT,
    ) -> "HeuristicPartitioner":
        """
        Initialize partitioner from files
        """

        return HeuristicPartitioner(
            state,
            *HeuristicPartitioner._read_files(state),
            alpha,
            slack_type,
            slack_value,
            max_iter,
        )

    def _solve_exact(
        self, G: Dict[int, List], P: Dict[int, int], D: Dict[Tuple[int, int], int], gap: float = 0.0
    ):
        """
        Solve districting problem on coarsened graph using exact method
        """
        ind_map = dict(zip(P.keys(), list(range(len(P)))))

        edges = {}
        for i in G:
            edges[ind_map[i]] = [ind_map[j] for j in G[i]]
        populations = list(P.values())
        distances = [[0] * len(P) for _ in range(len(P))]
        for i, j in D:
            distances[ind_map[i]][ind_map[j]] = D[i, j]

        partitioner = OptimalPartitioner(
            self.state,
            min(self.num_districts, len(P)),
            edges,
            populations,
            distances,
            self.alpha,
            self.slack_type,
            self.slack_value,
        )
        partitioner.optimize(gap)

        self.slack_value = partitioner.slack_value

        rev_map = {v: k for k, v in ind_map.items()}

        partitions = {}
        i = 0
        for partition in partitioner._get_district_counties().values():
            if len(partition) > 0:
                partitions[rev_map[i]] = [rev_map[j] for j in partition]
                i += 1

        return partitions

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
        for iii in range(self.max_iter):
            assert all(self._is_connected(partitions[i], G) for i in partitions)
            if skip == 0:
                sorted_border_nodes = sorted(border_nodes.items(), key=lambda x: x[1][0])
            i, new_part = sorted_border_nodes[skip][0]
            if sorted_border_nodes[skip][1][0] >= 0:
                break

            prev_part = node_partitions[i]
            if len(partitions[prev_part]) == 1:
                skip += 1
                continue

            partitions[prev_part].remove(i)
            partitions[new_part].append(i)
            # Second condition is magic bullshit. This should not be necessary.
            if not (
                self._is_connected(partitions[prev_part], G)
                and self._is_connected(partitions[new_part], G)
            ):
                partitions[prev_part].append(i)
                partitions[new_part].remove(i)
                skip += 1
                continue

            node_partitions[i] = new_part
            del border_nodes[i, new_part]
            skip = 0

            for node, part in border_nodes:
                node_part = node_partitions[node]
                if node_part in [prev_part, new_part] or part in [prev_part, new_part]:
                    _, count = border_nodes[node, part]
                    border_nodes[node, part] = (cost_increase(node, part), count)

            for j in G[i]:
                part_j = node_partitions[j]
                if new_part != part_j:
                    add_to_border(i, part_j)
                    add_to_border(j, new_part)
                else:
                    increase, count = border_nodes[j, prev_part]
                    if count == 1:
                        del border_nodes[j, prev_part]
                    else:
                        border_nodes[j, prev_part] = (increase, count - 1)

        return partitions

    def _optimize(
        self,
        G: Dict[int, List],
        P: Dict[int, int],
        D: Dict[Tuple[int, int], int],
        size_limit: int,
        gap: float = 0.0,
    ) -> Dict[int, List[int]]:
        if len(P) > size_limit:
            # Find maximal matching (heuristic)
            P_edges = sorted([(P[i] + P[j], i, j) for i in G for j in G[i] if i < j])

            matching = {}
            max_num_matching = 2 * (len(P) - size_limit)
            for _, i, j in P_edges:
                if i not in matching and j not in matching:
                    matching[i] = j
                    matching[j] = i
                if len(matching) >= max_num_matching:
                    break

            # Merge nodes in matching
            new_G = {}
            new_P = {}
            new_D = {}
            for i in P:
                if i not in matching:
                    new_P[i] = P[i]
                    new_G[i] = []
                elif i < matching[i]:
                    new_P[i] = P[i] + P[matching[i]]
                    new_G[i] = []

            map_to_match = lambda i: i if i not in matching or i < matching[i] else matching[i]

            for i in new_G:
                neighbor_set = set(G[i]) if i not in matching else set(G[i]) | set(G[matching[i]])
                new_G[i] = [k for k in {map_to_match(j) for j in neighbor_set} if k != i]

                distances = (
                    {j: D[i, j] for j in G if i != j}
                    if i not in matching
                    else {
                        j: (D[i, j] + D[matching[i], j]) / 2
                        for j in G
                        if i != j and matching[i] != j
                    }
                )

                for j in distances:
                    if j not in matching:
                        new_D[i, j] = new_D[j, i] = distances[j]
                    elif j < matching[j]:
                        new_D[i, j] = new_D[j, i] = distances[j] + distances[matching[j]] / 2

            sub_partitions = self._optimize(new_G, new_P, new_D, size_limit, gap)
            assert all(self._is_connected(sub_partitions[i], new_G) for i in sub_partitions)
            partitions = {i: [] for i in sub_partitions}

            # Unmerge nodes
            for i, partition in sub_partitions.items():
                for j in partition:
                    partitions[i].append(j)
                    if j in matching:
                        partitions[i].append(matching[j])

            # Local optimization
            return self._local_optimize(G, P, D, partitions)
        else:
            return self._solve_exact(G, P, D, gap)

    def optimize(self, gap: float = 0.0, size_limit=15):
        """
        Optimize partitioning using Swamy et al. (2022)
        """
        # TODO: who wrote this spaghetti code?
        G, P_list, D_mat, high_pop_inds = self.prepare_graph(remove_large_nodes=False)
        low_pop_inds = [i for i in range(self.num_counties) if i not in high_pop_inds]

        P = {i: p for i, p in zip(low_pop_inds, P_list)}
        D = {}
        for i, node_i in enumerate(low_pop_inds):
            for j, node_j in enumerate(low_pop_inds):
                D[node_i, node_j] = D[node_j, node_i] = D_mat[i][j]

        self.num_districts -= len(high_pop_inds)
        self.partitions = self._optimize(G, P, D, max(size_limit, self.num_districts), gap)

        # assert all(self._is_connected(self.partitions[i], G) for i in self.partitions)

        self.num_districts += len(high_pop_inds)
        self.partitions.update({i: [i] for i in high_pop_inds})

        return self.partitions

    def _get_district_counties(self) -> Dict[int, List[int]]:
        return self.partitions

    def print_solution(self) -> Dict[int, List[int]]:
        for i, partition in self.partitions.items():
            population = sum(self.populations[j] for j in partition)
            distance = sum(self.distances[j][k] for j in partition for k in partition if j != k) / 2
            value = (1 - self.alpha) * distance + self.C * self.alpha * abs(
                population - self.avg_population
            )

            print(f"District {i}:")
            print(partition)
            print(f"Population: {population}")
            print(f"Distance: {distance}")
            print(f"Value: {value}")
