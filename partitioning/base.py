import os
from typing import Dict, List, Tuple

from matplotlib.patches import Patch
import geopandas as gpd
import matplotlib.pyplot as plt

from settings.local_settings import DATA_PATH


class DistrictPartitioner:
    """
    Base class for district partitioning
    """

    def __init__(
        self, state: str, K: int, G: Dict[int, List], P: List[int], D: List[List[int]]
    ) -> None:
        """
        Initialize partitioner with state data

        @param state: State to partition
        @param K: Number of districts
        @param G: Adjacency list of counties
        @param P: Population of each county
        @param D: Distance between each pair of counties
        """

        self.state = state
        self.num_districts = K
        self.edges = G
        self.populations = P
        self.num_counties = len(P)
        self.total_population = sum(P)
        self.distances = D

    def _read_files(state: str) -> Tuple[int, Dict[int, List], List[int], List[List[int]]]:
        """
        Read data from files, including:
        - Population of each county
        - Distance between each pair of counties
        - Adjacency list of counties (for drawing map)
        - Number of districts
        """

        path = os.path.join(DATA_PATH, state, "counties", "graph")
        population_file = os.path.join(path, f"{state}.population")
        distances_file = os.path.join(path, f"{state}_distances.csv")
        neighbors_file = os.path.join(path, f"{state}.dimacs")

        with open(os.path.join(DATA_PATH, "Numberofdistricts.txt"), "r") as file:
            for line in file:
                parts = line.strip().split("\t")
                if parts[0] == state:
                    num_districts = int(parts[1])
                    break

        populations = []
        with open(population_file, "r") as file:
            total_population = int(next(file).split("=")[-1].strip())
            for line in file:
                ind, pop = line.split()
                populations.append(int(pop))
        num_counties = len(populations)

        distances = [[0] * num_counties for _ in range(num_counties)]
        edges = {i: [] for i in range(num_counties)}
        with open(neighbors_file, "r") as file:
            next(file)
            for line in file:
                if not line.startswith("e"):
                    break

                _, start, end = line.split()
                start = int(start)
                end = int(end)
                edges[start].append(end)
                edges[end].append(start)

        with open(distances_file, "r") as file:
            next(file)
            for i, line in enumerate(file):
                for j, val in enumerate(line.split(",")[1:]):
                    distances[i][j] = int(val)

        return num_districts, edges, populations, distances

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
        for district, counties in districts.items():
            for county in counties:
                counties_gdf.loc[counties_gdf["county_id"] == county, "district"] = district

        color_map = plt.get_cmap("tab20", self.num_districts)

        fig, ax = plt.subplots(figsize=(10, 10))
        counties_gdf.plot(
            column="district",
            ax=ax,
            categorical=True,
            legend=False,
            cmap=color_map,
        )

        district_pop = [sum(self.populations[j] for j in d) for d in districts.values()]
        district_dist = [
            sum(self.distances[i][j] for i in d for j in d if i != j) for d in districts.values()
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
