from typing import Dict, List, Tuple

from partitioning.base import DistrictPartitioner
from partitioning.swamy.optimal import OptimalPartitioner


class SwamyPartitioner(DistrictPartitioner):
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

    def __init__(self):
        """
        Initialize Swamy partitioner
        """

        super().__init__()

    def _solve_exact(
        self, G: Dict[int, List], P: List[int], D: Dict[Tuple[int, int], int], slack: float
    ):
        """
        Solve districting problem on coarsened graph using exact methods
        """

        partitioner = OptimalPartitioner(self.state, self.num_districts, G, P, D)
        partitioner.optimize()

        partitions = {}
        i = 0
        for partition in partitioner._get_district_counties().values():
            if len(partition) > 0:
                partitions[i] = partition
                i += 1

        return partitions

    def _optimize(
        self,
        G: Dict[int, List],
        P: List[int],
        D: Dict[Tuple[int, int], int],
        size_limit: int,
        slack: float,
    ) -> Dict[int, List]:
        if len(P) * self.num_districts > size_limit:
            # Find maximal matching (heuristic)
            P_edges = sorted([(P[i] + P[j], i, j) for i in G for j in G[i]])

            matching = {}
            for _, i, j in P_edges:
                if i not in matching and j not in matching:
                    matching[i] = j
                    matching[j] = i

            # Merge nodes in matching
            new_G = {}
            new_P = []
            new_D = {}
            for i in range(len(P)):
                if i not in matching:
                    new_G[i] = G[i].copy()
                    new_P.append(P[i])
                    for j in G:
                        if (i, j) in D:
                            new_D[j, i] = new_D[i, j] = D[i, j]
                else:
                    j = matching[i]
                    new_G[i] = list(set(G[i]) | set(G[j]))
                    new_P[i] = P[i] + P[j]
                    for k in G:
                        if (i, k) in D and (j, k) in D and k != i and k != j:
                            # Improve
                            new_D[i, k] = new_D[k, i] = (D[i, k] + D[j, k]) / 2

            sub_partitions = self._optimize(new_G, new_P, new_D, size_limit, slack)
            partitions = {i: [] for i in range(self.num_districts)}

            # Unmerge nodes
            for i, partition in sub_partitions.items():
                for j in partition:
                    partitions[i].append(j)
                    if j in matching:
                        partitions[i].append(matching[j])

            # Local optimization
            return self._local_optimize(G, P, D, partitions)
        else:
            return self._solve_exact(G, P, D, slack)

    def optimize(self, size_limit=50, slack=0.02):
        """
        Optimize partitioning using Swamy et al. (2022)
        """

        return self._optimize(self.edges, self.population, size_limit, slack)
        # coarse_graph = self._coarsen_graph()
        # coarse_partitioner = OptimalPartitioner(coarse_graph)
        # return self.unmerge(coarse_partitioner.optimize(), size_limit, slack)
