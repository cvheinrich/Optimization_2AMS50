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
        remove_large_nodes = self.slack_type == BSP.SLACK_FIXED
        G, P_list, D_mat, high_pop_inds = self.prepare_graph(remove_large_nodes)
        low_pop_inds = [i for i in range(self.num_counties) if i not in high_pop_inds]

        P = {i: p for i, p in zip(low_pop_inds, P_list)}
        D = {}
        for i, node_i in enumerate(low_pop_inds):
            for j, node_j in enumerate(low_pop_inds):
                D[node_i, node_j] = D[node_j, node_i] = D_mat[i][j]

        self.num_districts -= len(high_pop_inds)
        actual_size_limit = min(self.num_counties, max(size_limit, self.num_districts))
        self.partitions = self._optimize(G, P, D, actual_size_limit, gap)

        self.num_districts += len(high_pop_inds)
        self.partitions.update({i: [i] for i in high_pop_inds})

        return self.partitions

    def _get_district_counties(self) -> Dict[int, List[int]]:
        return self.partitions

    def _get_total_cost(self) -> float:
        return sum(self._get_partition_cost(partition) for partition in self.partitions.values())

    def print_solution(self) -> Dict[int, List[int]]:
        for i, partition in self.partitions.items():
            population = sum(self.populations[j] for j in partition)
            distance = sum(self.distances[j][k] for j in partition for k in partition if j != k) / 2
            cost = self._get_partition_cost(partition)

            print(f"\nDistrict {i}:")
            print(partition)
            print(f"Population: {population}")
            print(f"Distance: {distance}")
            print(f"Cost: {cost}")
