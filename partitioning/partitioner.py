import os

import gurobipy as gp
from gurobipy import GRB
import pymetis as metis
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt

from settings.local_settings import DATA_PATH


class DistrictPartitioner:
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

        self.population = {}
        with open(population_file, "r") as file:
            self.total_population = int(next(file).split("=")[-1].strip())
            for line in file:
                ind, pop = line.split()
                self.population[int(ind)] = int(pop)
        self.num_counties = len(self.population.keys())

        self.distances = {}
        with open(neighbors_file, "r") as file:
            next(file)
            for line in file:
                if not line.startswith("e"):
                    break

                _, start, end = line.split()
                start = int(start)
                end = int(end)
                self.distances[start, end] = 1

        matrix = [
            [0 for _ in range(self.num_counties)] for _ in range(self.num_counties)
        ]

        with open(distances_file, "r") as file:
            next(file)
            for i, line in enumerate(file):
                for j, val in enumerate(line.split(",")[1:]):
                    matrix[i][j] = val

        for i, j in self.distances:
            self.distances[i, j] = matrix[i][j]

    def __init__(self, state: str):
        self.state = state
        self._read_data()

    def optimize(self):
        raise NotImplementedError

    def print_solution(self):
        raise NotImplementedError

    def _get_district_counties(self):
        raise NotImplementedError

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


# ---------------------------------------------------#


class OptimalPartitioner(DistrictPartitioner):
    def _create_model(self, alpha: float):
        self.alpha = alpha
        self.model = gp.Model("model")

        self.x = {}
        self.source = {}
        for i in range(self.num_counties):
            for j in range(self.num_districts):
                self.x[i, j] = self.model.addVar(
                    vtype=gp.GRB.BINARY, name="x_{}_{}".format(i, j)
                )
                self.source[i, j] = self.model.addVar(
                    vtype=gp.GRB.BINARY, name="source_{}_{}".format(i, j)
                )

        self.y = {}
        self.flow = {}
        for i, k in self.distances:
            for j in range(self.num_districts):
                self.y[i, k, j] = self.model.addVar(
                    vtype=gp.GRB.BINARY, name="y_{}_{}_{}".format(i, k, j)
                )
                if i < k:
                    self.flow[i, k, j] = self.model.addVar(
                        vtype=gp.GRB.INTEGER, name="flow_{}_{}_{}".format(i, k, j)
                    )

        self.slack = {}
        for j in range(self.num_districts):
            self.slack[j] = self.model.addVar(vtype=gp.GRB.CONTINUOUS, name="slack")

        self.model.setObjective(
            gp.quicksum(
                self.y[i, k, j] * self.distances[i, k]
                for i, k in self.distances
                for j in range(self.num_districts)
            )
            + self.alpha
            * (gp.quicksum(self.slack[j] for j in range(self.num_districts))),
            gp.GRB.MINIMIZE,
        )

        for i, k in self.distances:
            for j in range(self.num_districts):
                # Edges can only be included if both endpoints are included
                self.model.addConstr(self.x[i, j] + self.x[k, j] <= 1 + self.y[i, k, j])
                # Edges must be included if both endpoints are included
                self.model.addConstr(self.x[i, j] + self.x[k, j] >= 2 * self.y[i, k, j])
                # Included edges must have flow
                # if i < k:
                #     self.model.addConstr(self.flow[i, k, j] >= self.y[i, k, j])

        for j in range(self.num_districts):
            self.model.addConstr(
                gp.quicksum(self.source[i, j] for i in range(self.num_counties)) == 1
            )
            for i in range(self.num_counties):
                self.model.addConstr(self.x[i, j] >= self.source[i, j])

                # Sources must have adequate outgoing flow
                print(i)
                out_neighbors = [
                    k
                    for k in range(i + 1, self.num_counties)
                    if self.distances.get((i, k), -1) != -1
                ]
                print(out_neighbors)
                self.model.addConstr(
                    self.source[i, j]
                    * (
                        gp.quicksum(self.x[k, j] for k in range(self.num_counties))
                        - gp.quicksum(self.flow[i, k, j] for k in out_neighbors)
                        - 1
                    )
                    == 0
                )
                # Non-sources must have adequate outgoing flow
                in_neighbors = [
                    k for k in range(i) if self.distances.get((k, i), -1) != -1
                ]
                print(in_neighbors)
                self.model.addConstr(
                    (1 - self.source[i, j])
                    * (
                        gp.quicksum(self.flow[k, i, j] for k in in_neighbors)
                        - gp.quicksum(self.flow[i, k, j] for k in out_neighbors)
                        - 1
                    )
                    == self.x[i, j] - 1
                )

        # Add population constraint with slack variable
        for j in range(self.num_districts):
            # Slack must put population within range
            self.model.addConstr(
                gp.quicksum(
                    self.x[i, j] * self.population[i] for i in range(self.num_counties)
                )
                >= ((self.total_population / self.num_districts) - self.slack[j])
            )
            self.model.addConstr(
                gp.quicksum(
                    self.x[i, j] * self.population[i] for i in range(self.num_counties)
                )
                <= ((self.total_population / self.num_districts) + self.slack[j])
            )

        # Each county is assigned to exactly one district
        for i in range(self.num_counties):
            self.model.addConstr(
                gp.quicksum(self.x[i, j] for j in range(self.num_districts)) == 1
            )

    def __init__(self, state: str, alpha: float):
        super().__init__(state)
        self._create_model(alpha)

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
                for j in range(self.num_districts):
                    if self.x[i, j].X > 0:
                        print("x_{}_{} = {}".format(i, j, self.x[i, j].X))
            for i, k in self.distances:
                for j in range(self.num_districts):
                    if self.y[i, k, j].X > 0:
                        print(f"y_{i}_{k}_{j} = {self.y[i, k, j].X}")
            for j in range(self.num_districts):
                print("slack_{} = {}".format(j, self.slack[j].X))
        else:
            print("No solution")

    def _get_district_counties(self):
        return [
            {i for i in range(self.num_counties) if self.x[i, j].X > 0.5}
            for j in range(self.num_districts)
        ]


# ---------------------------------------------------#


class MetisPartitioner(DistrictPartitioner):
    def __init__(self, state: str):
        super().__init__(state)
