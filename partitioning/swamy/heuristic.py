from typing import Dict, List, Tuple

from partitioning.base import DistrictPartitioner
from partitioning.swamy.optimal import OptimalPartitioner


class HeuristicPartitioner(DistrictPartitioner):
    """
    Steps of the partitioner as described in Swamy et al. (2022):
    1. Coarsen the graph:
        - Merge nodes until the graph is small enough to be solved by exact methods:
            - This can be done by iteratively finding maximal matchings or a maximum matchings.
            - To find compact maximum matchings:
                - Sort neighbor edges by the sum of populations of the two endpoints (adding
                the distance could also be considered)
                - Iterate over edges in non-decreasing order, add the current edge to the matching
                if it does not have an already included endpoint.
            - Merged on edges in the matching
    2. Solve districting problem on coarsened graph:
        - The exact method in `OptimalPartitioner` can be used
        - It is advantageous to start from an initial feasible solution:
            - Select random node not in a district
            - Attach random neighbor until the population exceeds the average
            - Repeat until all nodes are in a district
            - If K districts were found => done
            - If more than K => merge districts (possibly with the smallest populations)
            - If less than K => start over
        - The initial solution can be improved by the method described in 3. Uncoarsening
    3. Uncoarsen the graph:
        - Unmerge nodes to previous level (i.e. first undo the last matching merge, then the second
          to last, etc)
        - Use local improvement heuristic to improve solution:
            - The method used in the paper selects the county-neighboring district pair, that would
              improve the compactness objective the most, if the county was to be reassigned to the
              neighboring district.
            - Local search can be terminated after a certain number of iterations or when no improvement
              is found
            - Potentially useful data structures:
                - A data structure maintaining the list of counties neighboring other districts
                - A data structure maintaining the contribution of each county to the compactness objective
        - Repeat until the original graph is reached

    """

    def __init__(
        self,
        state: str,
        K: int,
        G: Dict[int, List[int]],
        P: List[int],
        D: List[List[int]],
        max_iter: int = 500,
        slack_value: float = 0.05,
    ):
        """
        Initialize Swamy partitioner
        """

        super().__init__(state, K, G, P, D)
        self.avg_population = self.total_population / K
        self.max_iter = max_iter
        self.slack = slack_value

    def from_files(
        state: str, max_iter: int = 500, slack_value: float = 0.05
    ) -> "HeuristicPartitioner":
        """
        Initialize partitioner from files
        """

        return HeuristicPartitioner(
            state, *DistrictPartitioner._read_files(state), max_iter, slack_value
        )

    def _solve_exact(self, G: Dict[int, List], P: Dict[int, int], D: Dict[Tuple[int, int], int]):
        """
        Solve districting problem on coarsened graph using exact methods
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
            slack_type=OptimalPartitioner.SLACK_DYNAMIC,
        )
        partitioner.optimize()

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
            abs(sum(P[i] for i in prev_part_nodes) - self.avg_population) < self.slack
            and abs(sum(P[i] for i in new_part_nodes) - self.avg_population) < self.slack
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
                if neighbor not in visited:
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

        def compactness_decrease(node: int, partition: int) -> int:
            dists_in_new = sum(D[node, i] for i in partitions[partition])
            dists_in_prev = sum(D[node, i] for i in partitions[node_partitions[node]] if i != node)
            return dists_in_new - dists_in_prev

        # key: (node, partition), value: (compactness decrease, number of neighbors in partition)
        border_nodes: Dict[Tuple[int, int], Tuple[int, int]] = {}

        def add_to_border(node, partition):
            if (node, partition) not in border_nodes:
                border_nodes[node, partition] = (compactness_decrease(node, partition), 1)
            else:
                increase, count = border_nodes[node, partition]
                border_nodes[node, partition] = (increase, count + 1)

        for i in G:
            part_i = node_partitions[i]
            for j in G[i]:
                part_j = node_partitions[j]
                if part_i != part_j:
                    add_to_border(i, part_j)

        for _ in range(self.max_iter):
            i, new_part = min(border_nodes, key=lambda x: border_nodes[x])
            if border_nodes[i, new_part][0] >= 0:
                break

            prev_part = node_partitions[i]
            partitions[prev_part].remove(i)
            if not self._is_connected(partitions[prev_part], G) or not self._is_balanced(
                partitions[prev_part], partitions[new_part], P
            ):
                partitions[prev_part].append(i)
                continue
            partitions[new_part].append(i)
            node_partitions[i] = new_part
            del border_nodes[i, new_part]

            for node, part in border_nodes:
                node_part = node_partitions[node]
                if node_part in [prev_part, new_part] or part in [prev_part, new_part]:
                    decrease, count = border_nodes[node, part]
                    border_nodes[node, part] = (compactness_decrease(node, part), count)

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

    def _optimize(
        self, G: Dict[int, List], P: Dict[int, int], D: Dict[Tuple[int, int], int], size_limit: int
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
                        new_D[i, j] = new_D[j, i] = (distances[j] + distances[matching[j]]) / 2

            sub_partitions = self._optimize(new_G, new_P, new_D, size_limit)
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
            return self._solve_exact(G, P, D)

    def optimize(self, size_limit=10):
        """
        Optimize partitioning using Swamy et al. (2022)
        """

        P = {i: p for i, p in enumerate(self.populations)}
        D = {}
        for i in range(len(self.distances)):
            for j in range(i + 1, len(self.distances[i])):
                D[i, j] = D[j, i] = self.distances[i][j]

        self.partitions = self._optimize(self.edges, P, D, max(size_limit, self.num_districts))

        return self.partitions

    def _get_district_counties(self) -> Dict[int, List[int]]:
        return self.partitions

    def print_solution(self) -> Dict[int, List[int]]:
        print(self.partitions)
