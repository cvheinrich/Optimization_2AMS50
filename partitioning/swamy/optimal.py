from typing import Dict, List
from partitioning.swamy.base import BaseSwamyPartitioner as BSP
import random

import gurobipy as gp
from gurobipy import GRB


class OptimalPartitioner(BSP):
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
    ) -> None:
        """
        Initialize optimal partitioner.

        @param state: State to partition
        @param alpha: Weight for population imbalance
        @param slack_type: Type of slack to use: "fixed", "var", or "dynamic"
        @param slack_value: Value of slack to use (irrelevant in case of slack_type="var")
        """
        super().__init__(state, K, G, P, D, alpha, slack_type, slack_value, max_iter)
        self._create_model()
        self._generate_initial_solution()

    def from_files(
        state: str,
        alpha: float = BSP.ALPHA_DEFAULT,
        slack_type: str = BSP.SLACK_DEFAULT,
        slack_value: float = BSP.SLACK_VALUE_DEFAULT,
        max_iter: int = BSP.MAX_ITER_DEFAULT,
    ) -> "OptimalPartitioner":
        """
        Initialize optimal partitioner from files
        """

        return OptimalPartitioner(
            state, *OptimalPartitioner._read_files(state), alpha, slack_type, slack_value, max_iter
        )

    def _create_model(self) -> None:
        """
        Create gurobi model for partitioning
        """
        self.model = gp.Model("model")

        self.x = {
            (i, j): self.model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}_{j}")
            for i in range(self.num_counties)
            for j in range(self.num_counties)
        }

        self.y = {
            (i, j, k): self.model.addVar(vtype=gp.GRB.BINARY, name=f"y_{i}_{j}_{k}")
            for i in range(self.num_counties)
            for j in range(self.num_counties)
            for k in range(j + 1, self.num_counties)
        }

        self.flow = {
            (i, j, k): self.model.addVar(vtype=gp.GRB.INTEGER, name=f"flow_{i}_{j}_{k}")
            for i in range(self.num_counties)
            for j in range(self.num_counties)
            for k in self.edges[j]
        }

        if self.slack_type == BSP.SLACK_VARIABLE:
            self.slack = [
                self.model.addVar(vtype=gp.GRB.CONTINUOUS) for _ in range(self.num_counties)
            ]
        else:  # BSP.SLACK_FIXED
            self.slack = [self.slack_value for _ in range(self.num_counties)]

        self.model.setObjective(
            (
                # Minimize within-district distances, i.e. spread
                gp.quicksum(
                    self.y[i, j, k] * self.distances[i][k]
                    for i in range(self.num_counties)
                    for j in range(self.num_counties)
                    for k in range(j + 1, self.num_counties)
                )
                # Minimize population imbalance
                + self.C
                * self.alpha
                * self.avg_population
                * gp.quicksum(self.slack[j] for j in range(self.num_counties))
            ),
            gp.GRB.MINIMIZE,
        )

        for i in range(self.num_counties):
            for j in range(self.num_counties):
                for k in range(j + 1, self.num_counties):
                    # If two endpoints are in the same district,
                    # the distance between them also has to be
                    self.model.addConstr(self.y[i, j, k] >= self.x[i, j] + self.x[i, k] - 1)

        for i in range(self.num_counties):
            # Population greater than lower bound
            self.model.addConstr(
                gp.quicksum(self.x[i, j] * self.populations[j] for j in range(self.num_counties))
                >= self.avg_population * self.x[i, i] * (1 - self.slack[i])
            )
            # Population less than upper bound
            self.model.addConstr(
                gp.quicksum(self.x[i, j] * self.populations[j] for j in range(self.num_counties))
                <= self.avg_population * self.x[i, i] * (1 + self.slack[i])
            )

        # Exactly num_districts district "centers"
        self.model.addConstr(
            gp.quicksum(self.x[i, i] for i in range(self.num_counties)) == self.num_districts
        )

        for j in range(self.num_counties):
            # Each county assigned to exactly one district
            self.model.addConstr(gp.quicksum(self.x[i, j] for i in range(self.num_counties)) == 1)
            for i in range(self.num_counties):
                # Counties are only assigned to district centers
                self.model.addConstr(self.x[i, j] <= self.x[i, i])

                if i != j:
                    # Non-centers consume adequate flow
                    self.model.addConstr(
                        self.x[i, j]
                        + gp.quicksum(
                            self.flow[i, j, k] - self.flow[i, k, j] for k in self.edges[j]
                        )
                        == 0
                    )
                else:
                    # District center has adequate outgoing flow
                    self.model.addConstr(
                        self.x[j, j]
                        + gp.quicksum(
                            self.flow[j, j, k] - self.flow[j, k, j] for k in self.edges[j]
                        )
                        - gp.quicksum(self.x[j, k] for k in range(self.num_counties))
                        == 0
                    )
                # Counties must be within district to receive flow
                self.model.addConstr(
                    self.num_counties * self.x[i, j]
                    - gp.quicksum(self.flow[i, k, j] for k in self.edges[j])
                    >= 0
                )

    def _get_initial_partition(self) -> List[List[int]]:
        partitions = []
        unpartitioned_nodes = set(range(self.num_counties))

        while len(unpartitioned_nodes) > 0:
            center = random.choice(tuple(unpartitioned_nodes))
            partition = [center]
            unpartitioned_nodes.remove(center)
            neighbors = {node for node in self.edges[center] if node in unpartitioned_nodes}
            partition_population = self.populations[center]

            while partition_population < self.avg_population and len(neighbors) > 0:
                node = random.choice(tuple(neighbors))
                partition.append(node)
                unpartitioned_nodes.remove(node)
                neighbors.remove(node)
                partition_population += self.populations[node]
                neighbors = neighbors | {
                    node for node in self.edges[node] if node in unpartitioned_nodes
                }
            partitions.append(partition)

        return partitions

    def _set_initial_flow(self, center: int, node: int, visited: List[bool]) -> int:
        visited[node] = True
        outflow = 0

        for neighbor in self.edges[node]:
            if self.x[center, neighbor].start == 1 and not visited[neighbor]:
                outflow_neighbor = self._set_initial_flow(center, neighbor, visited)
                self.flow[center, node, neighbor].start = outflow_neighbor
                outflow += outflow_neighbor

        return outflow + 1

    def _generate_initial_solution(self):
        partitions = []
        partition_pops = []
        while len(partitions) < self.num_districts:
            partitions = self._get_initial_partition()

        while len(partitions) > self.num_districts:
            smallest_partitions = []
            for _ in range(2):
                smallest = min(partitions, key=lambda x: sum(self.populations[i] for i in x))
                smallest_ind = partitions.index(smallest)
                smallest_partitions.append(partitions.pop(smallest_ind))
            partitions.append(smallest_partitions[0] + smallest_partitions[1])

        partitions = self._local_optimize(
            self.edges,
            {i: p for i, p in enumerate(self.populations)},
            {
                (i, j): d
                for i, row in enumerate(self.distances)
                for j, d in enumerate(row)
                if i != j
            },
            {i: l for i, l in enumerate(partitions)},
        )

        for i in range(self.num_counties):
            for j in range(self.num_counties):
                self.x[i, j].start = 0
                for k in range(j + 1, self.num_counties):
                    self.y[i, j, k].start = 0

        for partition in partitions.values():
            i = partition[0]
            self.x[i, i].start = 1
            for j in partition[1:]:
                self.x[i, j].start = 1

        self.model.update()

        for i in range(self.num_counties):
            if self.x[i, i].start == 1:
                self._set_initial_flow(i, i, [False] * self.num_counties)
                for j in range(self.num_counties):
                    for k in range(j + 1, self.num_counties):
                        if self.x[i, j].start == 1 and self.x[i, k].start == 1:
                            self.y[i, j, k].start = 1

        part_ind = 0
        for i in range(self.num_counties):
            if self.x[i, i].start == 1:
                population_ratio = (
                    sum(self.populations[j] for j in partitions[part_ind]) / self.avg_population
                )
                part_ind += 1
                lower_slack = 1 - population_ratio
                upper_slack = 1 + population_ratio
                self.slack[i].start = max(lower_slack, upper_slack)

        self.model.update()

    def update_model(self, alpha: float, slack_type: str, slack_value: float) -> None:
        """
        Update model with new parameters
        """
        self.alpha = alpha
        self.slack_type = slack_type
        self.slack_value = slack_value
        self._create_model()

    def optimize(self, gap=0, slack_step=0.001) -> gp.Model:
        """
        Optimize model.

        @param gap: MIP gap
        @param slack_step: Step size for increasing slack_value in case of infeasible model
        """
        self.model.Params.MIPGap = gap
        self.model.optimize()

        if self.slack_type == BSP.SLACK_FIXED and slack_step > 0:
            print("Increasing slack until feasible model is found. Set slack_step=0 to disable.")
            while self.model.status != GRB.OPTIMAL:
                self.slack_value += slack_step
                print(f"Slack: {self.slack_value}")
                self.update_model(self.alpha, self.slack_type, self.slack_value)
                self.model.optimize()

        if self.model.status == GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write("model.ilp")
        return self.model

    def print_solution(self) -> None:
        """
        Print solution of model
        """

        if self.model.status == GRB.OPTIMAL:
            print("\nObjective Value: %g" % self.model.ObjVal)
            for i in range(self.num_counties):
                for j in range(self.num_counties):
                    if self.x[i, j].X != 0:
                        print("x_{}_{} = {}".format(i, j, self.x[i, j].X))
            for i in range(self.num_districts):
                for j in range(self.num_counties):
                    for k in range(j + 1, self.num_counties):
                        if self.y[i, j, k].X != 0:
                            print(f"y_{i}_{k}_{j} = {self.y[i, j, k].X}")
            if self.slack_type == BSP.SLACK_VARIABLE:
                for j in range(self.num_districts):
                    print("slack_{} = {}".format(j, self.slack[j].X))
        else:
            print("No solution")

    def _get_district_counties(self) -> Dict[int, List[int]]:
        return {
            i: [j for j in range(self.num_counties) if self.x[i, j].X == 1]
            for i in range(self.num_counties)
            if self.x[i, i].X == 1
        }
