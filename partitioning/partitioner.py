import gurobipy as gp
from gurobipy import GRB
import pymetis as metis

from settings.local_settings import DATA_PATH


class DistrictPartitioner:
    def _read_data(self):
        path = f"{DATA_PATH}/{self.state}/counties/graph"
        population_file = f"{path}/{self.state}.population"
        distances_file = f"{path}/{self.state}_distances.csv"
        neighbors_file = f"{path}/{self.state}.dimacs"

        with open(f"{DATA_PATH}/Numberofdistricts.txt", "r") as file:
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


# ---------------------------------------------------#


class OptimalPartitioner(DistrictPartitioner):
    def _create_model(self, alpha: float):
        self.alpha = alpha
        self.model = gp.Model("model")

        self.x = {}
        for i in range(self.num_counties):
            for j in range(self.num_districts):
                self.x[i, j] = self.model.addVar(
                    vtype=gp.GRB.BINARY, name="x_{}_{}".format(i, j)
                )

        self.y = {}
        for i, k in self.distances:
            for j in range(self.num_districts):
                self.y[i, k, j] = self.model.addVar(
                    vtype=gp.GRB.BINARY, name="y_{}_{}_{}".format(i, k, j)
                )

        self.slack = {}
        for j in range(self.num_districts):
            self.slack[j] = self.model.addVar(vtype=gp.GRB.CONTINUOUS, name="slack")

        self.model.setObjective(
            -gp.quicksum(
                self.y[i, k, j] * self.distances[i, k]
                for i, k in self.distances
                for j in range(self.num_districts)
            )
            + self.alpha
            * (gp.quicksum(self.slack[j] for j in range(self.num_districts))),
            gp.GRB.MINIMIZE,
        )

        # Each edge in neighbors is assigned to exactly one district
        for i, k in self.distances:
            for j in range(self.num_districts):
                self.model.addConstr(self.x[i, j] + self.x[k, j] >= 2 * self.y[i, k, j])
                self.model.addConstr(self.x[i, j] + self.x[k, j] <= 1 + self.y[i, k, j])

        # Add population constraint with slack variable
        for j in range(self.num_districts):
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
        return self.model.optimize()

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


# ---------------------------------------------------#


class MetisPartitioner(DistrictPartitioner):
    def __init__(self, state: str):
        super().__init__(state)
