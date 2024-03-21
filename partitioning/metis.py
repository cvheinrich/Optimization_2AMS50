import pymetis as metis

from partitioning.base import DistrictPartitioner


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
        vweights = self.populations
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
