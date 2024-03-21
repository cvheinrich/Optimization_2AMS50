import argparse
from partitioning.partitioner import OptimalPartitioner as OP, MetisPartitioner

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
    parser.add_argument(
        "-l",
        "--slack_value",
        type=float,
        default=2.0,
        help="Starting slack value for fixed and dynamic solution types",
    )
    parser.add_argument(
        "-g", "--gap", type=float, default=0.0, help="Gap for optimal partitioner"
    )

    args = parser.parse_args()
    p_type = args.type

    if p_type == "metis":
        dp = MetisPartitioner(args.state)
    elif p_type in OP.SLACK_TYPES + ["optimal"]:
        slack_type = (
            p_type
            if p_type in [OP.SLACK_FIXED, OP.SLACK_DYNAMIC]
            else OP.SLACK_VARIABLE
        )
        dp = OP(
            args.state, args.alpha, slack_type=slack_type, slack_value=args.slack_value
        )
        dp.optimize(gap=args.gap, slack_step=0.1)

    if args.verbose:
        dp.print_solution()
    if args.map:
        dp.show_map()
