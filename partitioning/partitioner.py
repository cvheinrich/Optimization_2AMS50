import os
from typing import List, Dict

import gurobipy as gp
from gurobipy import GRB
from matplotlib.patches import Patch
import pymetis as metis
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from settings.local_settings import DATA_PATH


class DistrictPartitioner:
    """
    Base class for district partitioning
    """

    def __init__(self, state: str) -> None:
        """
        Initialize partitioner with state data

        @param state: State to partition
        """

        self.state = state
        self._read_data()

    def _read_data(self) -> None:
        """
        Read data from files, including:
        - Population of each county
        - Distance between each pair of counties
        - Adjacency list of counties (for drawing map)
        - Number of districts
        """

        path = os.path.join(DATA_PATH, self.state, "counties", "graph")
        population_file = os.path.join(path, f"{self.state}.population")
        distances_file = os.path.join(path, f"{self.state}_distances.csv")
        neighbors_file = os.path.join(path, f"{self.state}.dimacs")

        with open(os.path.join(DATA_PATH, "Numberofdistricts.txt"), "r") as file:
            for line in file:
                parts = line.strip().split("\t")
                if parts[0] == self.state:
                    self.num_districts = int(parts[1])
                    break

        self.population = []
        with open(population_file, "r") as file:
            self.total_population = int(next(file).split("=")[-1].strip())
            for line in file:
                ind, pop = line.split()
                self.population.append(int(pop))
        self.num_counties = len(self.population)

        self.distances = [[0] * self.num_counties for _ in range(self.num_counties)]
        self.edges = {i: [] for i in range(self.num_counties)}
        with open(neighbors_file, "r") as file:
            next(file)
            for line in file:
                if not line.startswith("e"):
                    break

                _, start, end = line.split()
                start = int(start)
                end = int(end)
                self.edges[start].append(end)
                self.edges[end].append(start)

        with open(distances_file, "r") as file:
            next(file)
            for i, line in enumerate(file):
                for j, val in enumerate(line.split(",")[1:]):
                    self.distances[i][j] = int(val)

    def show_map(self) -> None:
        """
        Show map of counties with districts colored
        """

        shape_file = os.path.join(
            DATA_PATH, self.state, "counties", "maps", f"{self.state}_counties.shp"
        )
        counties_gdf = gpd.read_file(shape_file)

        counties_gdf["county_id"] = range(len(counties_gdf))
        counties_gdf["district"] = -1

        districts = self._get_district_counties()
        for district, counties in enumerate(districts):
            for county in counties:
                counties_gdf.loc[counties_gdf["county_id"] == county, "district"] = (
                    district
                )

        color_map = plt.get_cmap("tab20", self.num_districts)

        fig, ax = plt.subplots(figsize=(10, 10))
        counties_gdf.plot(
            column="district",
            ax=ax,
            categorical=True,
            legend=False,
            cmap=color_map,
        )

        district_pop = [
            sum(self.population[j] for j in district) for district in districts
        ]
        district_dist = [
            sum(self.distances[i][j] for i in district for j in district)
            for district in districts
        ]
        legend_elements = [
            Patch(
                facecolor=color_map(district / self.num_districts),
                # edgecolor="k",
                label="Pop. {:,.0f}\nDist. {:,.0f}".format(
                    district_pop[district], district_dist[district]
                ),
            )
            for district in range(len(districts))
        ]
        ax.legend(
            handles=legend_elements,
            title="District Information",
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )

        plt.tight_layout()
        plt.show()

    def optimize(self):
        raise NotImplementedError

    def print_solution(self):
        raise NotImplementedError

    def _get_district_counties(self):
        raise NotImplementedError


# ---------------------------------------------------#


class OptimalPartitioner(DistrictPartitioner):
    SLACK_FIXED = "fixed"
    SLACK_VARIABLE = "var"
    SLACK_DYNAMIC = "dynamic"

    def __init__(
        self, state: str, alpha: float, slack_type="var", slack_value=None
    ) -> None:
        """
        Initialize optimal partitioner.

        @param state: State to partition
        @param alpha: Weight for population imbalance
        @param slack_type: Type of slack to use: "fixed", "var", or "dynamic"
        @param slack_value: Value of slack to use (irrelevant in case of slack_type="var")
        """

        super().__init__(state)
        self.alpha = alpha
        self.slack_type = slack_type
        self.slack_value = slack_value
        self.avg_population = self.total_population / self.num_districts
        self._create_model()

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

        if self.slack_type == OptimalPartitioner.SLACK_VARIABLE:
            self.slack = [
                self.model.addVar(vtype=gp.GRB.CONTINUOUS)
                for _ in range(self.num_counties)
            ]
        elif self.slack_type == OptimalPartitioner.SLACK_DYNAMIC:
            self.slack = [
                (
                    self.slack_value
                    if self.population[i] < self.avg_population
                    else self.population[i] / self.avg_population - 1 + self.slack_value
                )
                for i in range(self.num_counties)
            ]
        else:  # fixed
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
                    self.model.addConstr(
                        self.y[i, j, k] >= self.x[i, j] + self.x[i, k] - 1
                    )

        for i in range(self.num_counties):
            # Population greater than lower bound
            self.model.addConstr(
                gp.quicksum(
                    self.x[i, j] * self.population[j] for j in range(self.num_counties)
                )
                >= self.avg_population * self.x[i, i] * (1 - self.slack[i])
            )
            # Population less than upper bound
            self.model.addConstr(
                gp.quicksum(
                    self.x[i, j] * self.population[j] for j in range(self.num_counties)
                )
                <= self.avg_population * self.x[i, i] * (1 + self.slack[i])
            )

        # Exactly num_districts district "centers"
        self.model.addConstr(
            gp.quicksum(self.x[i, i] for i in range(self.num_counties))
            == self.num_districts
        )

        for j in range(self.num_counties):
            # Each county assigned to exactly one district
            self.model.addConstr(
                gp.quicksum(self.x[i, j] for i in range(self.num_counties)) == 1
            )
            for i in range(self.num_counties):
                # Counties are only assigned to district centers
                self.model.addConstr(self.x[i, j] <= self.x[i, i])

                if i != j:
                    # Non-centers consume adequate flow
                    self.model.addConstr(
                        self.x[i, j]
                        + gp.quicksum(
                            self.flow[i, j, k] - self.flow[i, k, j]
                            for k in self.edges[j]
                        )
                        == 0
                    )
                else:
                    # District center has adequate outgoing flow
                    self.model.addConstr(
                        self.x[j, j]
                        + gp.quicksum(
                            self.flow[j, j, k] - self.flow[j, k, j]
                            for k in self.edges[j]
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

    def optimize(self, gap=0, slack_step=0.1) -> gp.Model:
        """
        Optimize model.

        @param gap: MIP gap
        @param slack_step: Step size for increasing slack_value in case of infeasible model
        """
        self.model.Params.MIPGap = gap
        self.model.optimize()

        if (
            self.slack_type
            in [OptimalPartitioner.SLACK_FIXED, OptimalPartitioner.SLACK_DYNAMIC]
            and slack_step > 0
        ):
            print(
                "Increasing slack until feasible model is found. Set slack_step=0 to disable."
            )
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
            if self.slack_type == OptimalPartitioner.SLACK_VARIABLE:
                for j in range(self.num_districts):
                    print("slack_{} = {}".format(j, self.slack[j].X))
        else:
            print("No solution")

    def _get_district_counties(self) -> List[Dict]:
        return [
            [j for j in range(self.num_counties) if self.x[i, j].X > 0.5]
            for i in range(self.num_counties)
            if self.x[i, i].X > 0.5
        ]


# ---------------------------------------------------#


class MetisPartitioner(DistrictPartitioner):
    def __init__(self, state: str):
        """
        Initialize METIS partitioner
        """

        super().__init__(state)

    def optimize(self):
        """
        Optimize partitioning using METIS
        """

        n = self.num_counties
        xadj = [i for i in range(0, n * (n - 1), n - 1)]
        adjncy = [j for i in range(n) for j in range(n) if i != j]
        eweights = [self.distances[i][j] for i in range(n) for j in range(n) if i != j]
        vweights = self.population
        contiguous = True

        (edgecuts, part) = metis.part_graph(
            nparts=self.num_districts,
            xadj=xadj,
            adjncy=adjncy,
            eweights=eweights,
            vweights=vweights,
            contiguous=contiguous,
        )

        print("Edgecuts: ", edgecuts)
        print("Part: ", part)
