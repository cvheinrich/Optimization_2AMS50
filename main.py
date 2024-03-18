import argparse
from partitioning.partitioner import OptimalPartitioner, MetisPartitioner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Districting")
    parser.add_argument("-s", "--state", default="RI", help="State to district")
    parser.add_argument("-t", "--type", default="optimal", help="Type of partitioner")
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

    if args.type == "metis":
        dp = MetisPartitioner(args.state)
    else:
        dp = OptimalPartitioner(
            args.state, args.alpha, slack_type="dynamic", slack_value=0.35
        )

    dp.optimize(gap=0, slack_step=0.1)
    if args.verbose:
        dp.print_solution()
    if args.map:
        dp.show_map()
