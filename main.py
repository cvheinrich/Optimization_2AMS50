import sys
from partitioning.partitioner import OptimalPartitioner

if __name__ == "__main__":
    state = "RI"
    if len(sys.argv) > 1:
        state = sys.argv[1]

    dp = OptimalPartitioner(state, 1.0)
    dp.optimize()
    dp.print_solution()
