from typing import Dict, List, Tuple
from partitioning.swamy.base import BaseSwamyPartitioner as BSP

import gurobipy as gp
from gurobipy import GRB


class OptimalPartitioner(BSP):
    def __init__(
        self,
        state: str,
        K: int,
        G: Dict[int, List],
        P: List[int],
        D: List[List[int]],
        alpha: float = BSP.ALPHA_DEFAULT,
        slack_type: str = BSP.SLACK_DEFAULT,
        slack_value: float = BSP.SLACK_VALUE_DEFAULT,
    ) -> None:
        """
        Initialize optimal partitioner.

        @param state: State to partition
        @param alpha: Weight for population imbalance
        @param slack_type: Type of slack to use: "fixed", "var", or "dynamic"
        @param slack_value: Value of slack to use (irrelevant in case of slack_type="var")
        """
        super().__init__(state, K, G, P, D, alpha, slack_type, slack_value)
        self._create_model()

    def from_files(
        state: str,
        alpha: float = BSP.ALPHA_DEFAULT,
        slack_type: str = BSP.SLACK_DEFAULT,
        slack_value: float = BSP.SLACK_VALUE_DEFAULT,
    ) -> "OptimalPartitioner":
        """
        Initialize optimal partitioner from files
        """

        return OptimalPartitioner(
            state, *OptimalPartitioner._read_files(state), alpha, slack_type, slack_value
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
            # Minimize within-district distances
            (
                gp.quicksum(
                    self.y[i, j, k] * self.distances[i][k]
                    for i in range(self.num_counties)
                    for j in range(self.num_counties)
                    for k in range(j + 1, self.num_counties)
                )
                # Minimize population imbalance
                + self.alpha
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
        TODO: add large node isolation

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
            i: [j for j in range(self.num_counties) if self.x[i, j].X > 0.5]
            for i in range(self.num_counties)
            if self.x[i, i].X > 0.5
        }
