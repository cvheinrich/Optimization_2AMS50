import argparse
from partitioning.partitioner import OptimalPartitioner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Districting")
    parser.add_argument("-s", "--state", default="RI", help="State to district")
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=1.0,
        help="Alpha value for optimal partitioner",
    )
    parser.add_argument(
        "-m", "--map", type=bool, default=False, help="Show map of districts"
    )
    parser.add_argument(
        "-v", "--verbose", type=bool, default=False, help="Print solution"
    )

    args = parser.parse_args()

    dp = OptimalPartitioner(args.state, args.alpha)
    dp.optimize()
    if args.verbose:
        dp.print_solution()
    if args.map:
        dp.show_map()
