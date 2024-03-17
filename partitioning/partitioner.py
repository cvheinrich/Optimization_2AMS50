import os
import random
from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB
import pymetis as metis
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from settings.local_settings import DATA_PATH


class DistrictPartitioner:
    def __init__(self, state: str):
        self.state = state
        self._read_data()

    def _read_data(self):
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
                    self.distances[i][j] = val

    def show_map(self):
        shape_file = os.path.join(
            DATA_PATH, self.state, "counties", "maps", f"{self.state}_counties.shp"
        )
        counties_gdf = gpd.read_file(shape_file)

        counties_gdf["county_id"] = range(len(counties_gdf))
        counties_gdf["district"] = -1

        for district, counties in enumerate(self._get_district_counties()):
            for county in counties:
                counties_gdf.loc[counties_gdf["county_id"] == county, "district"] = (
                    district
                )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))
        counties_gdf.plot(
            column="district",
            ax=ax,
            legend=True,
            categorical=True,
            legend_kwds={"title": "District"},
        )
        plt.show()

    def optimize(self):
        raise NotImplementedError

    def print_solution(self):
        raise NotImplementedError

    def _get_district_counties(self):
        raise NotImplementedError


# ---------------------------------------------------#


class OptimalPartitioner(DistrictPartitioner):
    def __init__(self, state: str, alpha: float):
        super().__init__(state)
        self._create_model(alpha)

    def _create_model(self, alpha: float):
        self.alpha = alpha
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

        self.slack = [
            self.model.addVar(vtype=gp.GRB.CONTINUOUS) for _ in range(self.num_counties)
        ]

        avg_population = self.total_population / self.num_districts

        self.model.setObjective(
            # Minimize within-district distances
            gp.quicksum(
                self.y[i, j, k] * self.distances[i][k]
                for i in range(self.num_counties)
                for j in range(self.num_counties)
                for k in range(j + 1, self.num_counties)
            )
            # Minimize population imbalance
            + self.alpha
            * avg_population
            * gp.quicksum(self.slack[j] for j in range(self.num_counties)),
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
                >= avg_population * self.x[i, i] * (1 - self.slack[i])
            )
            # Population less than upper bound
            self.model.addConstr(
                gp.quicksum(
                    self.x[i, j] * self.population[j] for j in range(self.num_counties)
                )
                <= avg_population * self.x[i, i] * (1 + self.slack[i])
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

    def optimize(self):
        self.model.optimize()
        if self.model.status == GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write("model.ilp")
        return self.model

    def print_solution(self):
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
            for j in range(self.num_districts):
                print("slack_{} = {}".format(j, self.slack[j].X))
        else:
            print("No solution")

    def _get_district_counties(self):
        return [
            {j for j in range(self.num_counties) if self.x[i, j].X > 0.5}
            for i in range(self.num_counties)
        ]


# ---------------------------------------------------#


class MetisPartitioner(DistrictPartitioner):
    def __init__(self, state: str):
        super().__init__(state)
